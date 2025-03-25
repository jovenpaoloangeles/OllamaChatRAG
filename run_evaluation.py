#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Evaluation Pipeline Script

This script orchestrates the entire evaluation process:
1. Runs evaluate_models.py to evaluate Ollama models
2. Runs evaluate_api_models.py to evaluate API models (OpenAI, Gemini) if API keys are available
3. Processes the results with evaluation_analyzer.py
4. Creates visualizations using improve_visualization.py
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_command(command):
    """Run a command and return its output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    
    # Print output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    # Get return code
    return_code = process.poll()
    
    # Check if command was successful
    if return_code != 0:
        print(f"Error running command: {command}")
        error = process.stderr.read()
        print(f"Error message: {error}")
        return False
    
    return True

def get_latest_results_file():
    """Get the most recent evaluation results file"""
    results_dir = Path("evaluation_results")
    if not results_dir.exists():
        print("Error: evaluation_results directory not found")
        return None
    
    # Find all files that start with 'final_evaluation_results_'
    result_files = list(results_dir.glob("final_evaluation_results_*.json"))
    
    if not result_files:
        print("Error: No evaluation results files found")
        return None
    
    # Sort by modification time (most recent first)
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    # Return as a proper Path object string with forward slashes
    return str(latest_file).replace('\\', '/')

def modify_evaluation_analyzer(results_file):
    """Modify evaluation_analyzer.py to use the specified results file"""
    analyzer_path = Path("evaluation_analyzer.py")
    if not analyzer_path.exists():
        print("Error: evaluation_analyzer.py not found")
        return False
    
    # Read the original file
    with open(analyzer_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the line with file_path assignment
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "file_path =" in line and "'evaluation_results/" in line:
            # Replace the line with the new file path
            # Ensure we use forward slashes for the path
            normalized_path = results_file.replace('\\', '/')
            lines[i] = f"    file_path = '{normalized_path}'"
            break
    
    # Write the modified content back
    with open(analyzer_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return True

def check_api_keys(skip_api=False):
    """Check if API keys are available in the .env file"""
    has_api_keys = False
    
    # Don't check if skip_api is True
    if skip_api:
        return False
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("OpenAI API key found. Will include OpenAI models in evaluation.")
        has_api_keys = True
    
    # Check for Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        print("Gemini API key found. Will include Gemini models in evaluation.")
        has_api_keys = True
    
    return has_api_keys

def main():
    parser = argparse.ArgumentParser(description="Run the complete evaluation pipeline")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip running evaluate_models.py and use the latest results")
    parser.add_argument("--results-file", type=str, help="Specific results file to use (instead of the latest)")
    parser.add_argument("--skip-api", action="store_true", help="Skip API model evaluation even if API keys are available")
    args = parser.parse_args()
    
    # Check for API keys
    has_api_keys = check_api_keys(args.skip_api)
    
    # Step 1: Run evaluate_models.py for Ollama models (unless skipped)
    ollama_results_file = None
    if not args.skip_evaluation:
        print("\n===== Step 1: Running Ollama Model Evaluation =====")
        success = run_command("python evaluate_models.py")
        if not success:
            print("Error running evaluate_models.py. Exiting.")
            return
        
        # Get the latest results file after Ollama evaluation
        ollama_results_file = get_latest_results_file()
        if not ollama_results_file:
            print("Error getting latest results file after Ollama evaluation. Exiting.")
            return
    else:
        print("\n===== Step 1: Skipping Ollama Model Evaluation =====")
    
    # Step 2: Run evaluate_api_models.py for API models if API keys are available
    final_results_file = None
    if has_api_keys:
        print("\n===== Step 2: Running API Model Evaluation =====")
        
        # If we have Ollama results, pass them to the API evaluation
        api_command = "python evaluate_api_models.py"
        if ollama_results_file:
            api_command += f" --results-file {ollama_results_file}"
        elif args.results_file:
            api_command += f" --results-file {args.results_file}"
        
        success = run_command(api_command)
        if not success:
            print("Error running evaluate_api_models.py. Continuing with Ollama results only.")
            final_results_file = ollama_results_file or args.results_file or get_latest_results_file()
        else:
            # Get the latest results file after API evaluation
            final_results_file = get_latest_results_file()
    else:
        print("\n===== Step 2: Skipping API Model Evaluation (No API keys found or --skip-api specified) =====")
        # Use Ollama results or specified results file
        final_results_file = ollama_results_file or args.results_file or get_latest_results_file()
    
    # Step 3: Get the results file path
    if not final_results_file:
        print("Error: No results file available. Exiting.")
        return
    
    print(f"\nUsing results file: {final_results_file}")
    
    # Step 4: Modify evaluation_analyzer.py to use the specified results file
    print("\n===== Step 3: Processing Evaluation Results =====")
    if not modify_evaluation_analyzer(final_results_file):
        print("Error modifying evaluation_analyzer.py. Exiting.")
        return
    
    # Step 5: Run evaluation_analyzer.py to process results
    success = run_command("python evaluation_analyzer.py")
    if not success:
        print("Error running evaluation_analyzer.py. Exiting.")
        return
    
    # Step 6: Run improve_visualization.py for better visualizations
    print("\n===== Step 4: Generating Improved Visualizations =====")
    success = run_command("python improve_visualization.py")
    if not success:
        print("Error running improve_visualization.py. Exiting.")
        return
    
    print("\n===== Evaluation Pipeline Complete =====")
    print("Results processed and visualizations generated successfully!")
    print("\nVisualization files:")
    print("- accuracy_metrics.png - Shows Accuracy and Faithfulness metrics")
    print("- speed_metrics.png - Shows Response Time and Tokens per Second metrics")
    print("- answer_types.png - Shows distribution of answer types (Correct, Incorrect, Partial, No Information)")

if __name__ == "__main__":
    main()
