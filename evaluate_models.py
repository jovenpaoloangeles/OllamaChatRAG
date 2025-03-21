import os
import json
import subprocess
import time
from datetime import datetime
from tqdm import tqdm
import torch
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from create_vectorstore import index_exists
import requests
import logging

# Set logging level to reduce verbosity
logging.basicConfig(level=logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# Configuration
TEST_DATASET_PATH = "test_dataset.json"
OUTPUT_DIR = "evaluation_results"
USE_RERANKING = True
RERANK_TOP_K = 3
K_DOCS = 5
TEMPERATURE = 0.3

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to get available Ollama models
def get_available_models():
    try:
        # Run the ollama list command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        
        # Parse the output to extract model names
        lines = result.stdout.strip().split('\n')
        models = []
        
        # Skip the header line if it exists
        start_idx = 1 if lines and 'NAME' in lines[0] else 0
        
        for line in lines[start_idx:]:
            if line.strip():
                # Extract the model name (first column)
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    # Exclude nomic-embed-text as it doesn't support generate
                    if "nomic-embed-text" not in model_name:
                        models.append(model_name)
        
        return models if models else ["llama3.1", "gemma3:1b"]  # Fallback to default models
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}")
        return ["llama3.1", "gemma3:1b"]  # Fallback to default models

# Function to count tokens in a text string using Ollama API
def count_tokens(text, model="gemma3:1b"):
    try:
        # Since Ollama API tokenize endpoint may not be available in all versions,
        # we'll use a simple approximation method
        # For most modern LLMs, 1 token is roughly 4 characters for English text
        return len(text) // 4
        
        # The following code is commented out due to 404 errors with some Ollama versions
        # If your Ollama version supports the /api/tokenize endpoint, you can uncomment this
        """
        response = requests.post(
            "http://localhost:11434/api/tokenize",
            json={"model": model, "prompt": text}
        )
        
        if response.status_code == 200:
            tokens = response.json().get("tokens", [])
            return len(tokens)
        else:
            print(f"Error from Ollama API: {response.text}")
            # Fallback to approximate token count
            return len(text) // 4
        """
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        # Fallback to approximate token count
        return len(text) // 4

# Function to setup RAG pipeline
def setup_rag_pipeline(model_name):
    # Load embeddings and vectorstore
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    # Create base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": K_DOCS})
    
    # Apply reranking if enabled
    if USE_RERANKING:
        # Initialize the reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_K)
        
        # Create a contextual compression retriever with the reranker
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
    else:
        # Use the base retriever without reranking
        retriever = base_retriever
    
    # Create LLM with temperature setting
    llm = OllamaLLM(model=model_name, temperature=TEMPERATURE)
    
    # Create prompt template with more detailed instructions
    template = """
    
    You are a helpful assistant that answers questions based strictly on the provided context.

    Context:
    {context}

    Question: {question}

    Instructions:
    1. Only answer if the context provides a clear and complete answer.
    2. If the answer is not found in the context, respond with: "I don't have enough information to answer  this question."
    3. If the context contains partial or ambiguous information, provide a cautious partial answer starting     with <partial_answer>.
    4. Do NOT make assumptions or use any external knowledge.
    5. Keep the answer concise, clear, and well-formatted.
    
    Answer:
    """
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return qa_chain

# Main evaluation function
def evaluate_models():
    print("Starting evaluation...")
    
    # Check if index exists
    if not index_exists("chroma_db"):
        print("Error: Document index not found. Please run the app and index documents first.")
        return
    
    # Load test dataset
    try:
        with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
            test_dataset = json.load(f)
        print(f"Loaded {len(test_dataset)} test questions from {TEST_DATASET_PATH}")
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        return
    
    # Get available models
    try:
        available_models = get_available_models()
        print(f"Found {len(available_models)} available Ollama models: {', '.join(available_models)}")
    except Exception as e:
        print(f"Error getting available models: {str(e)}")
        return
    
    # Initialize results dictionary
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": available_models,
            "use_reranking": USE_RERANKING,
            "rerank_top_k": RERANK_TOP_K if USE_RERANKING else None,
            "k_docs": K_DOCS,
            "temperature": TEMPERATURE
        },
        "questions": []
    }
    
    # Process each question
    for i, item in enumerate(tqdm(test_dataset, desc="Processing questions", ncols=100, leave=True)):
        question = item["question"]
        true_answer = item["answer"]
        source = item.get("source", "Unknown")
        
        question_result = {
            "question": question,
            "true_answer": true_answer,
            "source": source,
            "model_answers": {}
        }
        
        # Get answers from each model
        for model_name in tqdm(available_models, desc=f"Question {i+1}/{len(test_dataset)}", ncols=100, leave=False):
            try:
                # Setup RAG pipeline for this model
                qa_chain = setup_rag_pipeline(model_name)
                
                # Get response
                start_time = time.time()
                response = qa_chain.invoke({"query": question})
                end_time = time.time()
                
                # Extract answer
                answer = response["result"]
                
                # Count tokens in the answer using the same model
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
                
                # Small delay to avoid overloading Ollama
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error with model {model_name} on question {i+1}: {str(e)}")
                question_result["model_answers"][model_name] = {
                    "answer": f"ERROR: {str(e)}",
                    "time_taken": None,
                    "token_count": 0,
                    "tokens_per_second": 0
                }
        
        # Add to results
        results["questions"].append(question_result)
        
        # Save intermediate results after each question
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"evaluation_results_{timestamp}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save final results
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_path = os.path.join(OUTPUT_DIR, f"final_evaluation_results_{final_timestamp}.json")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Results saved to {final_output_path}")
    print(f"Evaluated {len(test_dataset)} questions across {len(available_models)} models")

if __name__ == "__main__":
    evaluate_models()
