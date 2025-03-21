# RAG Chatbot with Reranking and Sentence Window Retrieval

This application is a Retrieval-Augmented Generation (RAG) chatbot that uses Ollama models to answer questions based on your documents. It features advanced retrieval techniques including reranking and sentence window chunking for improved accuracy.

## Features

### 1. Sentence Window Chunking

The application uses a two-step chunking process:
- First, documents are split into manageable chunks using RecursiveCharacterTextSplitter
- Then, a SentenceTransformersTokenTextSplitter is applied to create semantically meaningful chunks based on sentence boundaries

This approach helps maintain the context around sentences, improving the quality of retrieved information.

### 2. Reranking

The application uses CrossEncoderReranker to improve retrieval quality:
- Initial retrieval is done using vector similarity search
- Retrieved documents are then reranked using a cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
- Only the top-k reranked documents are used for generating the answer

Reranking helps prioritize the most relevant documents for the query, improving answer quality.

## Configuration Options

The application provides several configuration options in the sidebar:

### Model Settings
- **Ollama Model**: Choose between different Ollama models
- **Temperature**: Control the randomness of the model's output

### Retrieval Settings
- **Number of documents to retrieve**: Set how many documents to initially retrieve
- **Use reranking**: Toggle reranking on/off
- **Rerank top k**: When reranking is enabled, set how many documents to keep after reranking
- **Use sentence window chunking**: Toggle sentence window chunking for document indexing

## Usage

1. Add your PDF documents to the `data` directory
2. Click "Index Documents" in the sidebar
3. Once indexing is complete, ask questions in the chat input
4. View detailed retrieval information in the expandable sections

## Requirements

See requirements.txt for the full list of dependencies.

## Debug Information

The application provides several expandable sections for debugging:
- **Debug Information**: Shows basic information about the application state
- **Retrieval Debug**: Shows details about the retrieval process
- **Reranking Info**: Shows information about the reranking model when enabled
- **Retrieved Documents**: Shows the content and metadata of retrieved documents
