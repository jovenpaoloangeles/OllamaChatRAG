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

## Evaluation Pipeline

The application includes a comprehensive evaluation pipeline to assess model performance:

### Features

- **Multi-Model Evaluation**: Evaluate multiple models side-by-side (Ollama, OpenAI, Gemini)
- **Detailed Metrics**: Measures accuracy, faithfulness, response time, and tokens per second
- **Answer Categorization**: Classifies answers as correct, incorrect, partial, or no information
- **Visualization**: Generates clear visualizations for easy comparison

### Running Evaluations

1. Create a `.env` file with your API keys (copy from `.env.example`)
2. Run the evaluation pipeline:

```bash
python run_evaluation.py
```

### Command-Line Options

```bash
# Skip evaluation and use existing results
python run_evaluation.py --skip-evaluation

# Use a specific results file
python run_evaluation.py --results-file evaluation_results/your_results_file.json

# Keep temporary files for debugging
python run_evaluation.py --keep-temp
```

### Output

The evaluation generates three visualization files:
- `accuracy_metrics.png`: Shows Accuracy and Faithfulness metrics
- `speed_metrics.png`: Shows Response Time and Tokens per Second metrics
- `answer_types.png`: Shows distribution of answer types

### API Integration

The evaluation pipeline supports:
- **OpenAI Models**: GPT-3.5 Turbo and GPT-4
- **Google Gemini Models**: Gemini 1.5 Pro and Gemini 1.5 Flash

To use these models, add your API keys to the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## Non-UI RAG Pipeline

In addition to the Streamlit UI, this project includes a standalone RAG pipeline that can be used without a UI for integration into other applications or for command-line usage.

### Features

- **Modular Design**: Easily integrate the RAG pipeline into other applications
- **Command-line Interface**: Run queries directly from the command line
- **Interactive Mode**: Chat with your documents in a terminal interface
- **LazyGraphRAG Support**: Hierarchical summarization for improved retrieval quality
- **Future OpenAI Support**: Structure in place for easy integration with OpenAI models

### Usage

#### Basic Usage

```bash
# Answer a single question
python run_rag.py "What is machine learning?"

# Run in interactive mode
python run_rag.py --interactive

# Use LazyGraphRAG for improved retrieval
python run_rag.py --use-lazy-graph-rag "What is machine learning?"
```

#### Advanced Options

```bash
# Use a different LLM model
python run_rag.py --llm "gemma:7b" "What is deep learning?"

# Change retrieval parameters
python run_rag.py --k-docs 8 --rerank-top-k 5 "What is reinforcement learning?"

# Force reindexing of documents
python run_rag.py --force-reindex --interactive

# Disable reranking
python run_rag.py --no-reranking "What is neural networks?"

# Configure LazyGraphRAG parameters
python run_rag.py --use-lazy-graph-rag --graph-k 150 --graph-start-k 40 --graph-max-depth 4 "What is deep learning?"
```

### Programmatic Usage

You can also use the RAG pipeline in your own Python code:

```python
from rag_pipeline import RAGPipeline

# Create the RAG pipeline
pipeline = RAGPipeline(
    llm_model="llama3.1:latest",
    embedding_model="nomic-embed-text:latest",
    temperature=0.7,
    k_docs=5,
    use_reranking=True,
    rerank_top_k=3,
    use_lazy_graph_rag=True  # Enable LazyGraphRAG
)

# Initialize the pipeline
if not pipeline.vectorstore_ready:
    pipeline.index_documents()
pipeline.initialize_components()

# Ask a question
result = pipeline.answer_question("What is machine learning?")
print(result["answer"])
```

### LazyGraphRAG Implementation

The LazyGraphRAG approach improves retrieval quality through hierarchical summarization:

1. **Graph Traversal**: Retrieves a network of related documents using graph traversal
2. **Community Detection**: Identifies communities in the retrieved sub-graph
3. **Claim Extraction**: Extracts relevant claims from each community
4. **Claim Ranking**: Ranks claims based on relevance to the question
5. **Answer Generation**: Generates an answer based on the top-ranked claims

This approach often provides more comprehensive answers than traditional vector similarity retrieval, especially for complex questions that require synthesizing information from multiple sources.

### Future OpenAI Integration

The pipeline is designed to be extended with OpenAI models in the future. The structure is in place to add this functionality by setting up the appropriate API keys in your .env file.

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
