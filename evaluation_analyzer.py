import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import re
import pandas as pd
import os
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Groq API
def configure_groq():
    """Configure the Groq API with the API key"""
    api_key = "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"  # Using the provided API key
    client = groq.Client(api_key=api_key)
    return client

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load the evaluation results
def load_evaluation_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Clean text by removing <thought> blocks and other artifacts
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove <thought>...</thought> blocks
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove <partial_answer> tags
    text = re.sub(r'<partial_answer>|</partial_answer>', '', text)
    # Remove newlines
    text = re.sub(r'\n', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Initialize the sentence transformer model once as a global variable
# Using a smaller, more efficient model
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Calculate semantic similarity between model answer and true answer
def calculate_semantic_similarity(model_answer, true_answer):
    # Clean texts
    model_answer_clean = clean_text(model_answer)
    true_answer_clean = clean_text(true_answer)
    
    # If either answer is empty after cleaning, return 0
    if not model_answer_clean or not true_answer_clean:
        return 0.0
    
    # Encode the sentences to get their embeddings
    model_embedding = sentence_model.encode([model_answer_clean])
    true_embedding = sentence_model.encode([true_answer_clean])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(model_embedding, true_embedding)[0][0]
    return similarity

# Accuracy evaluation using Groq as judge
def evaluate_accuracy_with_groq(model_answer, true_answer, client, question):
    # Clean the answers
    model_answer_clean = clean_text(model_answer)
    true_answer_clean = clean_text(true_answer)
    
    # If the model answer is empty or indicates insufficient information, score it as 0
    if not model_answer_clean or "don't have enough information" in model_answer_clean.lower() or "insufficient information" in model_answer_clean.lower():
        return 0.0
    
    # Create a prompt for Groq to evaluate the answer
    prompt = f"""
    You are an expert evaluator for question-answering systems. Your task is to evaluate how accurately a model's answer matches the true answer to a question.
    
    Question: {question}
    True Answer: {true_answer_clean}
    Model Answer: {model_answer_clean}
    
    Please evaluate the model's answer on a scale from 0.0 to 1.0, where:
    - 0.0 means the answer is completely incorrect or unrelated to the true answer
    - 0.5 means the answer is partially correct but missing key information or contains inaccuracies
    - 1.0 means the answer is completely correct and covers all the key points in the true answer
    
    Return ONLY a number between 0.0 and 1.0 representing your evaluation score, with no explanation or additional text.
    """
    
    try:
        # Use Groq's LLama3-70b-8192 model
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Use low temperature for consistent evaluation
            max_tokens=10  # We only need a single number
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract the numerical score
        try:
            score = float(response_text)
            # Ensure the score is within the valid range
            score = max(0.0, min(1.0, score))
            return score
        except ValueError:
            # If Groq didn't return a valid number, fall back to semantic similarity
            print(f"Warning: Groq didn't return a valid score. Response: {response_text}. Falling back to semantic similarity.")
            return calculate_semantic_similarity(model_answer_clean, true_answer_clean)
    
    except Exception as e:
        print(f"Error using Groq for evaluation: {str(e)}. Falling back to semantic similarity.")
        return calculate_semantic_similarity(model_answer_clean, true_answer_clean)

# Categorize model answers into types
def categorize_answer(answer, true_answer, question, groq_client):
    if not isinstance(answer, str):
        return "no_information"
    
    # Clean the answer for checking
    clean_answer = clean_text(answer).lower()
    
    # Check for "I don't have enough information" responses
    if ("don't have enough information" in clean_answer or 
        "insufficient information" in clean_answer or
        "i don't have" in clean_answer and "information" in clean_answer):
        return "no_information"
    
    # Check for partial answers
    if "<partial_answer>" in answer or "partial answer" in clean_answer:
        return "partial"
    
    # If it's a complete answer, evaluate if it's correct or incorrect
    # Use Groq to evaluate accuracy
    accuracy = evaluate_accuracy_with_groq(answer, true_answer, groq_client, question)
    
    # If accuracy is high, consider it correct, otherwise incorrect
    if accuracy >= 0.7:  # Threshold for correctness
        return "correct"
    else:
        return "incorrect"

# Count answer types for each model
def count_answer_types(results, groq_client):
    answer_types = {}
    
    for question in results['questions']:
        true_answer = question['true_answer']
        question_text = question['question']
        
        for model, data in question['model_answers'].items():
            if model not in answer_types:
                answer_types[model] = {
                    "correct": 0,
                    "incorrect": 0,
                    "partial": 0,
                    "no_information": 0
                }
            
            answer = data.get('answer', '')
            answer_type = categorize_answer(answer, true_answer, question_text, groq_client)
            
            if answer_type in answer_types[model]:
                answer_types[model][answer_type] += 1
    
    return answer_types

# Compute average time taken per model
def compute_average_time(results):
    time_per_model = {}
    count_per_model = {}
    
    for question in results['questions']:
        for model, data in question['model_answers'].items():
            if 'time_taken' in data:
                time_per_model[model] = time_per_model.get(model, 0) + data['time_taken']
                count_per_model[model] = count_per_model.get(model, 0) + 1
    
    avg_time_per_model = {model: time_per_model[model] / count_per_model[model] for model in time_per_model}
    return avg_time_per_model

# Compute average tokens per second per model
def compute_average_tokens_per_second(results):
    tokens_per_second_per_model = {}
    count_per_model = {}
    
    for question in results['questions']:
        for model, data in question['model_answers'].items():
            if 'tokens_per_second' in data and data['tokens_per_second'] is not None:
                tokens_per_second_per_model[model] = tokens_per_second_per_model.get(model, 0) + data['tokens_per_second']
                count_per_model[model] = count_per_model.get(model, 0) + 1
    
    avg_tokens_per_second = {model: tokens_per_second_per_model[model] / count_per_model[model] 
                            for model in tokens_per_second_per_model}
    return avg_tokens_per_second

# Process all evaluation results
def process_evaluation_results(results):
    # Initialize dictionaries to store results
    faithfulness_scores = {}
    accuracy_scores = {}
    
    # Configure Groq for LLM-as-Judge evaluation
    groq_client = configure_groq()
    
    # Get total number of questions and models for progress reporting
    total_questions = len(results['questions'])
    question_count = 0
    
    # Process each question
    for question in results['questions']:
        question_count += 1
        true_answer = question['true_answer']
        question_text = question['question']
        
        print(f"Processing question {question_count}/{total_questions}: {question_text[:50]}...")
        
        for model, data in question['model_answers'].items():
            if model not in faithfulness_scores:
                faithfulness_scores[model] = []
                accuracy_scores[model] = []
            
            model_answer = data.get('answer', '')
            
            # Calculate faithfulness score (semantic similarity)
            faithfulness = calculate_semantic_similarity(model_answer, true_answer)
            faithfulness_scores[model].append(faithfulness)
            
            # Calculate accuracy score using Groq as judge
            accuracy = evaluate_accuracy_with_groq(model_answer, true_answer, groq_client, question_text)
            accuracy_scores[model].append(accuracy)
            
            print(f"  - Evaluated {model}: Faithfulness={faithfulness:.2f}, Accuracy={accuracy:.2f}")
    
    # Calculate average scores for each model
    avg_faithfulness = {model: np.mean(scores) for model, scores in faithfulness_scores.items()}
    avg_accuracy = {model: np.mean(scores) for model, scores in accuracy_scores.items()}
    
    # Calculate average time taken per model
    avg_time = compute_average_time(results)
    
    # Calculate average tokens per second per model
    avg_tokens_per_second = compute_average_tokens_per_second(results)
    
    # Count answer types
    answer_types = count_answer_types(results, groq_client)
    
    return {
        'faithfulness': avg_faithfulness,
        'accuracy': avg_accuracy,
        'time_taken': avg_time,
        'tokens_per_second': avg_tokens_per_second,
        'answer_types': answer_types
    }

# Generate visualizations
def generate_visualizations(processed_results):
    models = list(processed_results['faithfulness'].keys())
    
    # Create a figure with 5 subplots (added tokens per second and answer types)
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    
    # 1. Faithfulness Plot
    faithfulness_values = [processed_results['faithfulness'][model] for model in models]
    axes[0].bar(models, faithfulness_values)
    axes[0].set_title('Faithfulness (Semantic Similarity)')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Accuracy Plot
    accuracy_values = [processed_results['accuracy'][model] for model in models]
    axes[1].bar(models, accuracy_values)
    axes[1].set_title('Accuracy (Groq LLama3-70b as Judge)')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Average Time Plot
    time_values = [processed_results['time_taken'][model] for model in models]
    axes[2].bar(models, time_values)
    axes[2].set_title('Average Time Taken (seconds)')
    axes[2].set_xlabel('Models')
    axes[2].set_ylabel('Time (s)')
    axes[2].tick_params(axis='x', rotation=45)
    
    # 4. Tokens Per Second Plot
    tokens_per_second_values = [processed_results['tokens_per_second'][model] for model in models]
    axes[3].bar(models, tokens_per_second_values)
    axes[3].set_title('Tokens Per Second')
    axes[3].set_xlabel('Models')
    axes[3].set_ylabel('Tokens/s')
    axes[3].tick_params(axis='x', rotation=45)
    
    # 5. Answer Types Stacked Bar Chart
    correct_counts = [processed_results['answer_types'][model]['correct'] for model in models]
    incorrect_counts = [processed_results['answer_types'][model]['incorrect'] for model in models]
    partial_counts = [processed_results['answer_types'][model]['partial'] for model in models]
    no_info_counts = [processed_results['answer_types'][model]['no_information'] for model in models]
    
    # Create stacked bar chart
    width = 0.8
    axes[4].bar(models, correct_counts, width, label='Correct Answers', color='#2ca02c')
    
    # Calculate the bottom positions for each segment
    bottom_for_incorrect = correct_counts.copy()
    axes[4].bar(models, incorrect_counts, width, bottom=bottom_for_incorrect, label='Incorrect Answers', color='#d62728')
    
    bottom_for_partial = [correct + incorrect for correct, incorrect in zip(correct_counts, incorrect_counts)]
    axes[4].bar(models, partial_counts, width, bottom=bottom_for_partial, label='Partial Answers', color='#ff7f0e')
    
    bottom_for_no_info = [correct + incorrect + partial for correct, incorrect, partial in zip(correct_counts, incorrect_counts, partial_counts)]
    axes[4].bar(models, no_info_counts, width, bottom=bottom_for_no_info, label='No Information', color='#7f7f7f')
    
    axes[4].set_title('Answer Types by Model')
    axes[4].set_xlabel('Models')
    axes[4].set_ylabel('Number of Responses')
    axes[4].tick_params(axis='x', rotation=45)
    axes[4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a table summary
    data = {
        'Model': models,
        'Faithfulness': faithfulness_values,
        'Accuracy': accuracy_values,
        'Avg. Time (s)': time_values,
        'Tokens/s': tokens_per_second_values,
        'Correct Answers': correct_counts,
        'Incorrect Answers': incorrect_counts,
        'Partial Answers': partial_counts,
        'No Information': no_info_counts
    }
    
    df = pd.DataFrame(data)
    # Sort by accuracy (primary) and faithfulness (secondary)
    df_sorted = df.sort_values(by=['Accuracy', 'Faithfulness'], ascending=False)
    
    return df_sorted

def main():
    # File path
    file_path = 'evaluation_results/final_evaluation_results_20250321_091625.json'
    
    # Load the evaluation results
    results = load_evaluation_results(file_path)
    
    # Process the results
    processed_results = process_evaluation_results(results)
    
    # Generate visualizations
    summary_df = generate_visualizations(processed_results)
    
    # Save summary to CSV
    summary_df.to_csv('evaluation_summary.csv', index=False)
    
    # Print summary
    print("\nEvaluation Summary:")
    pd.set_option('display.precision', 4)
    print(summary_df.to_string(index=False))
    
    print("\nVisualizations saved to 'evaluation_results_visualization.png'")
    print("Summary saved to 'evaluation_summary.csv'")

if __name__ == "__main__":
    main()
