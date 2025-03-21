import os
import json
import argparse
from pathlib import Path
import google.generativeai as genai
from langchain_community.document_loaders import UnstructuredFileLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
def configure_genai(api_key):
    """Configure the Gemini API with the provided API key"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('models/gemini-2.0-flash')

# Custom loader class with better error handling
class SafeUnstructuredFileLoader(UnstructuredFileLoader):
    def load(self):
        try:
            return super().load()
        except Exception as e:
            logger.error(f"Error loading file {self.file_path}: {str(e)}")
            return []

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    try:
        logger.info(f"Loading {pdf_path}")
        loader = SafeUnstructuredFileLoader(pdf_path)
        docs = loader.load()
        if not docs:
            logger.warning(f"No content extracted from {pdf_path}")
            return ""
        
        # Combine all document chunks into a single text
        text = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def generate_questions(model, pdf_content, pdf_name, num_questions=5):
    """Generate questions and answers based on PDF content using Gemini"""
    if not pdf_content.strip():
        logger.warning(f"Empty content for {pdf_name}, skipping question generation")
        return []
    
    # Truncate content if it's too long (Gemini has context limits)
    max_content_length = 100000  # Adjust based on model's context window
    if len(pdf_content) > max_content_length:
        logger.warning(f"Content too long ({len(pdf_content)} chars), truncating to {max_content_length} chars")
        pdf_content = pdf_content[:max_content_length]
    
    prompt = f"""
    You are an expert at creating question-answer pairs for testing retrieval-augmented generation (RAG) systems.
    
    I will provide you with the content of a PDF document. Your task is to:
    1. Create {num_questions} diverse question-answer pairs based on the content.
    2. Focus on questions that test different aspects of information retrieval and comprehension.
    3. Include ONLY simple factual questions.
    4. Make sure each question has a clear, concise answer that can be directly found in or inferred from the document.
    5. Format your response as a JSON array of objects, each with 'question' and 'answer' fields.
    
    Document content:
    {pdf_content}
    
    Return ONLY the JSON array without any additional text or explanation.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response (handling potential formatting issues)
        try:
            # Try to parse the response directly
            qa_pairs = json.loads(response_text)
            
            # If the response is not a list, check if it's embedded in a code block
            if not isinstance(qa_pairs, list):
                # Look for JSON array in markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(.+?)\s*```', response_text, re.DOTALL)
                if json_match:
                    qa_pairs = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the text
            import re
            json_match = re.search(r'\[\s*{.+}\s*\]', response_text, re.DOTALL)
            if json_match:
                qa_pairs = json.loads(json_match.group(0))
            else:
                logger.error(f"Failed to parse JSON from response for {pdf_name}")
                return []
        
        # Validate the structure of qa_pairs
        if not isinstance(qa_pairs, list):
            logger.error(f"Expected list of QA pairs but got {type(qa_pairs)} for {pdf_name}")
            return []
        
        # Add source information to each QA pair
        for qa_pair in qa_pairs:
            qa_pair['source'] = pdf_name
        
        logger.info(f"Generated {len(qa_pairs)} QA pairs for {pdf_name}")
        return qa_pairs
    
    except Exception as e:
        logger.error(f"Error generating questions for {pdf_name}: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Generate test dataset for RAG from PDF files')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing PDF files')
    parser.add_argument('--output_file', type=str, default='test_dataset.json', help='Output JSON file for the dataset')
    parser.add_argument('--api_key', type=str, help='Gemini API key')
    parser.add_argument('--questions_per_pdf', type=int, default=2, help='Number of questions to generate per PDF')
    args = parser.parse_args()
    
    # Check if API key is provided
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.error("Gemini API key is required. Provide it with --api_key or set GEMINI_API_KEY environment variable.")
        return
    
    # Configure Gemini API
    model = configure_genai(api_key)
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    # Find all PDF files
    pdf_files = list(data_dir.glob('**/*.pdf'))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Generate questions for each PDF
    all_qa_pairs = []
    for pdf_file in pdf_files:
        pdf_content = extract_text_from_pdf(pdf_file)
        if pdf_content:
            qa_pairs = generate_questions(model, pdf_content, pdf_file.name, args.questions_per_pdf)
            all_qa_pairs.extend(qa_pairs)
    
    # Save the dataset
    output_file = Path(args.output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated dataset with {len(all_qa_pairs)} QA pairs, saved to {output_file}")

if __name__ == "__main__":
    main()
