# RAG Test Dataset Generator

This tool generates a test dataset for Retrieval-Augmented Generation (RAG) systems by creating question-answer pairs from PDF documents using Google's Gemini API.

## Prerequisites

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Obtain a Gemini API key from the [Google AI Studio](https://ai.google.dev/).

## Usage

### Basic Usage

Run the script with your Gemini API key:

```bash
python generate_test_dataset.py --api_key AIzaSyAjHVGHAwSvrT0oK4Gzj_ZY3PluelH3Kbs
```

This will:
1. Process all PDF files in the `data` directory
2. Generate 5 questions per PDF
3. Save the dataset to `test_dataset.json`

### Advanced Options

```bash
python generate_test_dataset.py \
  --data_dir path/to/pdfs \
  --output_file custom_dataset.json \
  --questions_per_pdf 10 \
  --api_key YOUR_API_KEY
```

You can also set your API key as an environment variable:

```bash
# Set environment variable
set GEMINI_API_KEY=your_api_key

# Run the script without passing the key
python generate_test_dataset.py
```

## Output Format

The generated dataset is a JSON file containing an array of question-answer pairs, each with:

```json
[
  {
    "question": "What is RAG?",
    "answer": "RAG (Retrieval-Augmented Generation) is a technique that enhances large language models by retrieving relevant information from external knowledge sources before generating responses.",
    "source": "example.pdf"
  },
  ...
]
```

## Tips for Better Results

1. **Organize your PDFs**: Place related documents in the same directory for better organization.
2. **Check PDF quality**: Make sure your PDFs have proper text extraction (not scanned images without OCR).
3. **Review the generated questions**: The quality of questions may vary based on the content and structure of your PDFs.
4. **Adjust questions per PDF**: For longer documents, you might want to increase the number of questions.

## Troubleshooting

- If you encounter errors related to PDF extraction, ensure you have all the necessary dependencies for the `unstructured` library.
- For large PDFs, the content might be truncated due to Gemini API's context window limitations.
