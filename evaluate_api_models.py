#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate API Models Script

This script evaluates API-based models (OpenAI and Gemini) using the same test dataset
as the main evaluation pipeline, and appends the results to an existing evaluation file.
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import logging

# Set logging level to reduce verbosity
logging.basicConfig(level=logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# API-based model imports
import google.generativeai as genai
import openai

# Load environment variables from .env file
load_dotenv()

# Configuration
TEST_DATASET_PATH = "test_dataset.json"
OUTPUT_DIR = "evaluation_results"
TEMPERATURE = 0.3

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to count tokens in a text string
def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        if "gpt" in model.lower():
            # Use tiktoken for OpenAI models
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            except ImportError:
                # Fallback to approximate token count
                return len(text) // 4
        else:
            # For Gemini models, use a simple approximation
            return len(text) // 4
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        # Fallback to approximate token count
        return len(text) // 4

# Function to get available API models
def get_available_api_models():
    api_models = []
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("OpenAI API key found. Including OpenAI models in evaluation.")
        openai.api_key = openai_api_key
        api_models.extend(["gpt-3.5-turbo", "gpt-4"])
    
    # Check for Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        print("Gemini API key found. Including Gemini models in evaluation.")
        genai.configure(api_key=gemini_api_key)
        api_models.extend(["models/gemini-2.0-flash"])
    
    return api_models

# Function to get answer from OpenAI model
def get_openai_answer(question, model_name):
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on your knowledge."},
                {"role": "user", "content": question}
            ],
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with OpenAI model {model_name}: {str(e)}")
        return f"ERROR: {str(e)}"

# Function to get answer from Gemini model
def get_gemini_answer(question, model_name):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        print(f"Error with Gemini model {model_name}: {str(e)}")
        return f"ERROR: {str(e)}"

# Function to get the latest evaluation results file
def get_latest_results_file():
    results_dir = Path(OUTPUT_DIR)
    if not results_dir.exists():
        print(f"Error: {OUTPUT_DIR} directory not found")
        return None
    
    # Find all files that start with 'final_evaluation_results_'
    result_files = list(results_dir.glob("final_evaluation_results_*.json"))
    
    if not result_files:
        print("Error: No evaluation results files found")
        return None
    
    # Sort by modification time (most recent first)
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    return str(latest_file).replace('\\', '/')

# Main evaluation function
def evaluate_api_models(results_file=None):
    print("Starting API model evaluation...")
    
    # Load test dataset
    try:
        with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
        print(f"Loaded {len(test_dataset)} test questions from {TEST_DATASET_PATH}")
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        return
    
    # Get available API models
    api_models = get_available_api_models()
    if not api_models:
        print("No API models available. Please check your .env file for API keys.")
        return
    
    print(f"Found {len(api_models)} available API models: {', '.join(api_models)}")
    
    # Load existing results if a file is specified
    if results_file and os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Loaded existing results from {results_file}")
            
            # Add API models to metadata
            results["metadata"]["api_models"] = api_models
            
        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
            # Initialize new results if there's an error loading the file
            results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "api_models": api_models,
                    "temperature": TEMPERATURE
                },
                "questions": []
            }
    else:
        # Initialize new results dictionary
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "api_models": api_models,
                "temperature": TEMPERATURE
            },
            "questions": []
        }
    
    # Process each question
    for i, item in enumerate(tqdm(test_dataset, desc="Processing questions", ncols=100, leave=True)):
        question = item["question"]
        true_answer = item["answer"]
        source = item.get("source", "Unknown")
        
        # Find or create question result
        if results_file:
            # Find existing question in results
            question_result = None
            for q in results["questions"]:
                if q["question"] == question:
                    question_result = q
                    break
            
            if not question_result:
                # Create new question result if not found
                question_result = {
                    "question": question,
                    "true_answer": true_answer,
                    "source": source,
                    "model_answers": {}
                }
                results["questions"].append(question_result)
        else:
            # Create new question result
            question_result = {
                "question": question,
                "true_answer": true_answer,
                "source": source,
                "model_answers": {}
            }
            results["questions"].append(question_result)
        
        # Get answers from each API model
        for model_name in tqdm(api_models, desc=f"Question {i+1}/{len(test_dataset)}", ncols=100, leave=False):
            try:
                # Get response based on model type
                start_time = time.time()
                
                if "gpt" in model_name:
                    answer = get_openai_answer(question, model_name)
                elif "gemini" in model_name:
                    answer = get_gemini_answer(question, model_name)
                else:
                    raise ValueError(f"Unknown API model: {model_name}")
                
                end_time = time.time()
                
                # Count tokens in the answer
                token_count = count_tokens(answer, model=model_name)
                time_taken = end_time - start_time
                
                # Calculate tokens per second
                tokens_per_second = token_count / time_taken if time_taken > 0 else 0
                
                # Store result
                question_result["model_answers"][model_name] = {
                    "answer": answer,
                    "time_taken": time_taken,
                    "token_count": token_count,
                    "tokens_per_second": tokens_per_second
                }
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error with model {model_name} on question {i+1}: {str(e)}")
                question_result["model_answers"][model_name] = {
                    "answer": f"ERROR: {str(e)}",
                    "time_taken": None,
                    "token_count": 0,
                    "tokens_per_second": 0
                }
        
        # Save intermediate results after each question
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"api_evaluation_results_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save final results
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_path = os.path.join(OUTPUT_DIR, f"final_evaluation_results_{final_timestamp}.json")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Results saved to {final_output_path}")
    print(f"Evaluated {len(test_dataset)} questions across {len(api_models)} API models")
    
    return final_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate API models and append results to existing evaluation")
    parser.add_argument("--results-file", type=str, help="Existing results file to append to")
    args = parser.parse_args()
    
    # If no results file is specified, try to get the latest one
    results_file = args.results_file
    if not results_file:
        results_file = get_latest_results_file()
        if results_file:
            print(f"Using latest results file: {results_file}")
    
    evaluate_api_models(results_file)
