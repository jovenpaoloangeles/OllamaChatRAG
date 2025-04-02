# evaluate_rag_llm_judge_v3.py
# Simplified script to evaluate a ConfigurableRAGPipeline using:
# - Test Dataset: test_senate.json
# - Chunking Strategy: recursive
# - Indexing: Force reindex
# - Evaluation Method: llm_judge (Gemini)
# - Loads GEMINI_API_KEY from .env
# - Handles Gemini Rate Limits (429) with retries
# - Summarizes correct answers and average judge score
# - Improved source filename extraction

import os
import json
import argparse
import time
import re
import string
import logging
import random
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted # For rate limit handling
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv # Import dotenv

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file into environment

# Assuming configurable_rag.py containing ConfigurableRAGPipeline class is in the same directory
# or accessible via Python path.
try:
    from configurable_rag import ConfigurableRAGPipeline
except ImportError:
    print("Error: Failed to import ConfigurableRAGPipeline from configurable_rag.py.")
    print("Ensure configurable_rag.py is in the same directory or accessible in your PYTHONPATH.")
    exit(1)

# --- Configuration (Hardcoded based on user request) ---
TEST_DATASET_PATH = "test_senate.json"
CHUNKING_STRATEGY = "recursive"
FORCE_REINDEX = True
EVAL_METHOD = "llm_judge"
DEFAULT_OUTPUT_FILE = f"evaluation_results_{CHUNKING_STRATEGY}_{EVAL_METHOD}.json"
MAX_GEMINI_RETRIES = 5
GEMINI_BASE_WAIT_TIME = 2 # seconds

# --- Set up Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def load_test_data(filepath):
    """Loads the test dataset from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file should contain a list of objects.")
        logger.info(f"Successfully loaded {len(data)} items from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"Error: Test dataset file not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error: Could not decode JSON from {filepath}: {e}")
        return None
    except ValueError as e:
        logger.error(f"Error: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}")
        return None

def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation, extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove all punctuation
    text = ' '.join(text.split()) # Remove extra whitespace
    return text

# Initialize a sentence transformer model globally for semantic similarity fallback
try:
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence Transformer model 'all-MiniLM-L6-v2' loaded for semantic similarity fallback.")
except Exception as e:
    logger.warning(f"Could not load Sentence Transformer model: {e}. Semantic similarity fallback will not work.")
    similarity_model = None

def calculate_semantic_similarity(text1, text2):
    """Calculates cosine similarity between two texts using sentence-transformers (Fallback)."""
    if not similarity_model:
        logger.warning("Similarity model not loaded. Returning 0.0 for semantic similarity.")
        return 0.0
    if not text1 or not text2:
        return 0.0
    try:
        embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
        similarity = cosine_scores.item()
        similarity = max(0.0, min(1.0, (similarity + 1) / 2)) # Normalize -1..1 to 0..1
        return similarity
    except Exception as e:
        logger.error(f"Error calculating semantic similarity: {e}")
        return 0.0

def is_refusal(answer_text):
    """Check if the generated answer is an explicit refusal."""
    if not answer_text:
        return False
    normalized_answer = clean_text(answer_text)
    refusal_patterns = [
        r"i (don'?t|do not) know",
        r"i (don'?t|do not) have enough information",
        r"i am unable to answer",
        r"i cannot answer",
        r"no information provided",
        r"information is not available",
        r"not mentioned in the provided context",
        r"context does not contain",
        r"insufficient information",
    ]
    for pattern in refusal_patterns:
        if re.search(pattern, normalized_answer, re.IGNORECASE):
            return True
    return False

# Define the default prompt template for the LLM Judge
DEFAULT_JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator for question-answering systems. Your task is to evaluate how accurately a model's answer matches the true answer for the given question.
Focus ONLY on factual accuracy and completeness relative to the true answer. Ignore differences in phrasing or verbosity if the core information is correct.

Question: {question}
True Answer: {true_answer_clean}
Model Answer: {model_answer_clean}

Evaluate the Model Answer based on the True Answer on a CONTINOUS scale from 0.0 to 1.0:
- 0.0: Completely incorrect, irrelevant, or contains significant factual errors compared to the True Answer.
- 0.5: Partially correct, but misses key information from the True Answer or includes minor inaccuracies.
- 1.0: Completely correct and accurately covers all the essential information present in the True Answer.

Return ONLY the numerical score (e.g., 0.8). Do not include any explanation, symbols, or other text.
"""

def evaluate_accuracy_with_gemini(model_answer, true_answer, gemini_model, question, judge_prompt_template):
    """
    Evaluates the accuracy of a model's answer against a true answer using a Gemini model.
    Includes retry logic for rate limits (HTTP 429) and falls back to semantic similarity.

    Returns:
        float: A score between 0.0 and 1.0 indicating accuracy, or -1.0 on critical error (e.g., empty true answer),
               or the semantic similarity score if Gemini fails after retries or returns unusable output.
    """
    model_answer_clean = clean_text(model_answer)
    true_answer_clean = clean_text(true_answer)

    if not true_answer_clean:
         logger.warning("True answer is empty after cleaning. Cannot evaluate with LLM Judge.")
         return -1.0 # Indicate critical inability to judge
    if not model_answer_clean or is_refusal(model_answer):
        logger.info("Model answer is empty or a refusal. LLM Judge scoring 0.0.")
        return 0.0

    try:
        prompt = judge_prompt_template.format(
            question=question,
            true_answer_clean=true_answer_clean,
            model_answer_clean=model_answer_clean
        )
    except KeyError as e:
        logger.error(f"Judge prompt template missing key: {e}. Template: '{judge_prompt_template}'")
        return -1.0 # Critical error

    generation_config = genai.GenerationConfig(
        temperature=0.0,
        max_output_tokens=10,
        candidate_count=1
    )
    safety_settings = [
         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    response = None
    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            break # Success

        except ResourceExhausted as e:
            if attempt < MAX_GEMINI_RETRIES - 1:
                wait_time = (GEMINI_BASE_WAIT_TIME * (2 ** attempt)) + random.uniform(0, 1)
                logger.warning(f"Gemini rate limit hit (Attempt {attempt + 1}/{MAX_GEMINI_RETRIES}). Retrying in {wait_time:.2f} seconds. Error: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Gemini rate limit hit after {MAX_GEMINI_RETRIES} attempts. Falling back to semantic similarity. Last Error: {e}")
                return calculate_semantic_similarity(model_answer_clean, true_answer_clean)

        except Exception as e:
            logger.error(f"Error using Gemini for evaluation (Attempt {attempt + 1}): {e}. Falling back to semantic similarity.", exc_info=False)
            return calculate_semantic_similarity(model_answer_clean, true_answer_clean)

    # --- Process the response if successful ---
    if response is None:
         logger.error("Gemini evaluation failed after retries, fallback should have been returned.")
         return calculate_semantic_similarity(model_answer_clean, true_answer_clean) # Fallback just in case

    if not response.candidates:
        block_reason = response.prompt_feedback.block_reason
        logger.warning(f"Gemini judge response blocked due to safety settings. Reason: {block_reason}. Falling back to semantic similarity.")
        return calculate_semantic_similarity(model_answer_clean, true_answer_clean)

    response_text = response.text.strip()
    logger.debug(f"Gemini Judge Raw Response: '{response_text}'")

    try:
        score_match = re.search(r"(\d\.?\d*)", response_text)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score)) # Clamp score
            logger.info(f"Gemini Judge Score: {score:.3f}")
            return score
        else:
            logger.warning(f"Gemini judge didn't return a parseable number. Response: '{response_text}'. Falling back to semantic similarity.")
            return calculate_semantic_similarity(model_answer_clean, true_answer_clean)

    except ValueError:
        logger.warning(f"Gemini judge returned non-numeric text. Response: '{response_text}'. Falling back to semantic similarity.")
        return calculate_semantic_similarity(model_answer_clean, true_answer_clean)


def configure_genai(api_key, model_name='gemini-2.0-flash-lite'):
    """Configure the Gemini API and return the model."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini model '{model_name}' configured successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to configure Gemini model '{model_name}': {e}", exc_info=True)
        return None

# --- Main Evaluation Function ---
def main():
    parser = argparse.ArgumentParser(description=f"Evaluate RAG pipeline (Recursive Chunking, LLM Judge) on {TEST_DATASET_PATH}")

    # --- Core RAG Configuration Arguments ---
    parser.add_argument("--llm", type=str, default="llama3.1:latest", help="Ollama LLM model identifier")
    parser.add_argument("--embeddings", type=str, default="nomic-embed-text:latest", help="Ollama embedding model identifier")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM generation")

    # --- Retrieval Configuration ---
    parser.add_argument("--k-docs", type=int, default=10, help="Number of documents to initially retrieve")
    parser.add_argument("--no-reranking", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--rerank-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="HuggingFace model for reranker")
    parser.add_argument("--rerank-top-k", type=int, default=5, help="Number of documents to keep after reranking")

    # --- Indexing Configuration (Recursive Specific) ---
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size for 'recursive' strategy")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for 'recursive' strategy")

    # --- Data/Index Arguments ---
    parser.add_argument("--data-dir", default="data", help="Directory containing documents")
    parser.add_argument("--persist-dir", default="chroma_db", help="Directory to persist the vector store")

    # --- Evaluation Output and Judge Configuration ---
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="File to save evaluation results")
    parser.add_argument("--judge-threshold", type=float, default=0.79,
                        help="Score threshold for LLM judge to consider an answer correct (>= threshold)")
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY"),
                        help="Gemini API Key for LLM judge (reads from .env or override here). REQUIRED.")
    parser.add_argument("--judge-model-name", default="gemini-2.0-flash-lite",
                        help="Gemini model to use for LLM judge")

    args = parser.parse_args()

    # --- Initialize Judge ---
    if not args.gemini_api_key:
        logger.error("Gemini API key is required. Set GEMINI_API_KEY in .env file or provide --gemini-api-key.")
        return
    gemini_judge_model = configure_genai(args.gemini_api_key, args.judge_model_name)
    if not gemini_judge_model:
        logger.error("Failed to initialize Gemini judge model. Exiting.")
        return
    logger.info(f"Using LLM Judge ({args.judge_model_name}) for evaluation with threshold {args.judge_threshold}")

    # --- Load Test Data ---
    logger.info(f"Loading test data from '{TEST_DATASET_PATH}'...")
    test_data = load_test_data(TEST_DATASET_PATH)
    if test_data is None: return

    # --- Initialize RAG Pipeline ---
    logger.info(f"Initializing RAG pipeline with '{CHUNKING_STRATEGY}' chunking...")
    logger.info(f"  LLM: {args.llm}, Embeddings: {args.embeddings}")
    try:
        pipeline = ConfigurableRAGPipeline(
            llm_model=args.llm,
            embedding_model=args.embeddings,
            temperature=args.temperature,
            k_docs=args.k_docs,
            use_reranking=not args.no_reranking,
            rerank_model=args.rerank_model,
            rerank_top_k=args.rerank_top_k,
            chunking_strategy=CHUNKING_STRATEGY,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            data_dir=args.data_dir,
            persist_dir=args.persist_dir
         )
        logger.info("RAG pipeline object created.")
    except Exception as e:
        logger.error(f"Error initializing RAGPipeline: {e}", exc_info=True)
        return

    # --- Indexing (Forced) ---
    logger.info(f"Indexing documents (force_reindex={FORCE_REINDEX}). Using strategy: {pipeline.chunking_strategy}")
    try:
        indexing_result = pipeline.index_documents(force_reindex=FORCE_REINDEX)
        logger.info(f"Indexing result: {indexing_result}")

        logger.info("Initializing pipeline components...")
        if not pipeline.initialize_components():
            logger.error("Failed to initialize RAG pipeline components. Exiting.")
            return
    except Exception as e:
        logger.error(f"Error during indexing or component initialization: {e}", exc_info=True)
        return

    logger.info("RAG pipeline ready for evaluation.")

    # --- Evaluation Loop ---
    results_list = []
    total_rag_time = 0
    num_questions = len(test_data)
    start_eval_time = time.time()
    status_counts = {}
    total_llm_judge_score = 0.0 # <-- Initialize score accumulator
    llm_judge_evaluated_count = 0 # <-- Initialize count for averaging

    for i, item in enumerate(test_data):
        question = item.get("question")
        expected_answer = item.get("answer")
        expected_source = item.get("source")
        status = "unknown"
        generated_answer = "N/A"
        retrieved_sources = []
        source_names = []
        error_message = None
        duration = 0
        answer_match = None # Boolean
        llm_judge_score = None # Float
        source_match = None # Boolean

        if not question:
            logger.warning(f"Skipping item {i+1}/{num_questions}: Missing 'question' field.")
            status = "skipped"
            results_list.append({
                "index": i, "question": None, "status": status, "reason": "Missing question field"
            })
            status_counts[status] = status_counts.get(status, 0) + 1
            continue

        logger.info(f"\n[{i+1}/{num_questions}] Processing question: {question}")
        item_start_time = time.time()

        try:
            # --- Run RAG Pipeline ---
            result = pipeline.answer_question(question)
            item_end_time = time.time()
            duration = item_end_time - item_start_time
            total_rag_time += duration

            generated_answer = result.get("answer", "N/A")
            retrieved_sources = result.get("sources", [])

            # --- Extract Source Names (Improved) ---
            source_names = []
            for src in retrieved_sources:
                source_name = "Unknown"
                if isinstance(src, dict):
                    # Direct access to the source field for dictionary format
                    source_name = src.get("source", "Unknown")
                elif hasattr(src, "metadata") and isinstance(src.metadata, dict):
                    # Handle Document objects
                    source_name = src.metadata.get("source", "Unknown")
                elif isinstance(src, str):
                    source_name = src

                # Extract basename if it's a path
                if source_name != "Unknown":
                    source_names.append(os.path.basename(str(source_name)))
                else:
                    source_names.append("Unknown")

            # --- Evaluate Answer Accuracy (LLM Judge Only) ---
            if is_refusal(generated_answer):
                status = "refusal"
                answer_match = False
                llm_judge_score = 0.0 # Refusals score 0
            elif not expected_answer:
                 status = "no_expected_answer"
                 answer_match = None
                 llm_judge_score = None
            else:
                llm_judge_score = evaluate_accuracy_with_gemini(
                    model_answer=generated_answer,
                    true_answer=expected_answer,
                    gemini_model=gemini_judge_model,
                    question=question,
                    judge_prompt_template=DEFAULT_JUDGE_PROMPT_TEMPLATE
                )
                if llm_judge_score < 0.0: # Check for critical judge error signal (-1.0)
                     status = "judge_error"
                     answer_match = None
                     # Do not include this score in the average
                else:
                     answer_match = llm_judge_score >= args.judge_threshold
                     status = "evaluated" # Mark as evaluated, final status determined later
                     # Accumulate score for averaging if valid
                     total_llm_judge_score += llm_judge_score
                     llm_judge_evaluated_count += 1


            # --- Evaluate Source Match ---
            if expected_source:
                expected_source_basename = os.path.basename(expected_source)
                source_match = expected_source_basename in source_names
            else:
                source_match = None # Cannot evaluate source match

            # --- Determine Final Status ---
            if status == "evaluated":
                 answer_ok = answer_match is True
                 source_ok = source_match is True or source_match is None # OK if matched or no expectation

                 if answer_ok and source_ok:
                     status = "success"
                 elif not answer_ok and source_ok:
                     status = "answer_mismatch"
                 elif answer_ok and not source_ok:
                     status = "source_mismatch"
                 elif not answer_ok and not source_ok:
                     status = "mismatch_both"

            # --- Log Results for this item ---
            logger.info(f"  Generated Answer: {generated_answer}")
            if expected_answer: logger.info(f"  Expected Answer:  {expected_answer}")
            logger.info(f"  Retrieved Sources: {source_names}")
            if expected_source: logger.info(f"  Expected Source:   {expected_source}")
            logger.info(f"  RAG Time: {duration:.2f} seconds")
            logger.info(f"  Status: {status}")
            if llm_judge_score is not None and llm_judge_score >= 0.0: # Only log valid scores
                 logger.info(f"  LLM Judge Score: {llm_judge_score:.3f} (Threshold: {args.judge_threshold})")
            if answer_match is not None: logger.info(f"  Answer Match (Judge): {answer_match}")
            if source_match is not None: logger.info(f"  Source Match: {source_match}")

        except Exception as e:
            item_end_time = time.time()
            duration = item_end_time - item_start_time if duration == 0 else duration
            error_message = str(e)
            status = "error"
            logger.error(f"  Error processing question: {e}", exc_info=True)
            logger.info(f"  Approx. Time: {duration:.2f} seconds")
            logger.info(f"  Status: {status}")

        # --- Store Detailed Results ---
        results_list.append({
            "index": i,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "expected_source": expected_source,
            "retrieved_sources": source_names,
            "rag_time_seconds": round(duration, 2),
            "status": status,
            "evaluation_method": EVAL_METHOD if status not in ["skipped", "error", "refusal", "no_expected_answer", "judge_error"] else None,
            "answer_match": answer_match,
            # Store score even if it's a fallback score, but handle None/error signal (-1.0)
            "llm_judge_score": llm_judge_score if llm_judge_score is not None and llm_judge_score >= 0.0 else None,
            "judge_threshold": args.judge_threshold,
            "source_match": source_match,
            "error_message": error_message
        })
        status_counts[status] = status_counts.get(status, 0) + 1

    # --- Summary and Saving ---
    end_eval_time = time.time()
    total_script_duration = end_eval_time - start_eval_time

    # Calculate correct answers count from results
    correct_answers_count = sum(1 for r in results_list if r.get("answer_match") is True)

    # Calculate average LLM judge score
    average_llm_judge_score = None
    if llm_judge_evaluated_count > 0:
        average_llm_judge_score = total_llm_judge_score / llm_judge_evaluated_count

    logger.info(f"\n--- Evaluation Summary ---")
    logger.info(f"Dataset: {TEST_DATASET_PATH}")
    logger.info(f"RAG Pipeline Config:")
    logger.info(f"  LLM: {args.llm}, Embeddings: {args.embeddings}")
    logger.info(f"  Retrieve K Docs: {pipeline.k_docs}")
    logger.info(f"  Use Reranking: {pipeline.use_reranking}")
    if pipeline.use_reranking:
        logger.info(f"    Reranker Model: {pipeline.rerank_model}")
        logger.info(f"    Rerank Top K: {pipeline.rerank_top_k}")
    logger.info(f"  Chunking Strategy: {pipeline.chunking_strategy}")
    logger.info(f"    Recursive Chunk Size: {pipeline.chunk_size}, Overlap: {pipeline.chunk_overlap}")
    logger.info(f"Evaluation Config:")
    logger.info(f"  Method: {EVAL_METHOD} (Judge: {args.judge_model_name}, Threshold: {args.judge_threshold})")
    logger.info(f"Results:")
    logger.info(f"  Total questions in dataset: {num_questions}")
    logger.info("  Status Counts:")
    for S, count in sorted(status_counts.items()):
        logger.info(f"    - {S}: {count}")

    processed_count = num_questions - status_counts.get("skipped", 0)
    # Use llm_judge_evaluated_count for accuracy denominator if available, else processed_count
    eval_denominator = llm_judge_evaluated_count if llm_judge_evaluated_count > 0 else processed_count
    logger.info(f"  Correct Answers (Answer Match = True): {correct_answers_count} / {eval_denominator} questions evaluated by judge")
    if average_llm_judge_score is not None:
        logger.info(f"  Average LLM Judge Score: {average_llm_judge_score:.3f} (over {llm_judge_evaluated_count} questions)")
    else:
        logger.info(f"  Average LLM Judge Score: N/A (no questions successfully evaluated by judge)")
    logger.info(f"  Total RAG processing time: {total_rag_time:.2f} seconds.")
    logger.info(f"  Total script execution time: {total_script_duration:.2f} seconds.")
    avg_rag_time = total_rag_time / processed_count if processed_count > 0 else 0
    logger.info(f"  Average RAG time per question processed: {avg_rag_time:.2f} seconds.")

    # --- Save results to JSON file ---
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir)
                 logger.info(f"Created output directory: {output_dir}")
             except OSError as e:
                 logger.error(f"Could not create output directory {output_dir}: {e}")

        try:
            # Create a dictionary to save, including args for context
            output_data = {
                "evaluation_config": {
                    "test_dataset": TEST_DATASET_PATH,
                    "chunking_strategy": CHUNKING_STRATEGY,
                    "force_reindex": FORCE_REINDEX,
                    "eval_method": EVAL_METHOD,
                    "judge_model_name": args.judge_model_name,
                    "judge_threshold": args.judge_threshold,
                    **vars(args) # Include other runtime args like llm, embeddings, etc.
                },
                "summary": {
                    "total_questions": num_questions,
                    "processed_questions": processed_count,
                    "status_counts": status_counts,
                    "correct_answers_count": correct_answers_count,
                    "llm_judge_evaluated_count": llm_judge_evaluated_count, # Added count used for avg
                    "average_llm_judge_score": round(average_llm_judge_score, 3) if average_llm_judge_score is not None else None, # Added avg score
                    "total_rag_time_seconds": round(total_rag_time, 2),
                    "total_script_time_seconds": round(total_script_duration, 2),
                    "average_rag_time_per_question_seconds": round(avg_rag_time, 2),
                },
                "results": results_list
            }
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to '{args.output_file}'")
        except IOError as e:
            logger.error(f"Error saving results to '{args.output_file}': {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving results: {e}", exc_info=True)

if __name__ == "__main__":
    main()