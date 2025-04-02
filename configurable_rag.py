# configurable_rag.py
import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

# Ensure langchain_experimental is installed: pip install langchain-experimental
try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    print("Error: SemanticChunker requires 'langchain-experimental'.")
    print("Please install it: pip install langchain-experimental")
    sys.exit(1)

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.schema.retriever import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import numpy as np # SemanticChunker might use numpy
from pathlib import Path # For better path handling and globbing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom loader class with better error handling
class SafeUnstructuredFileLoader(UnstructuredFileLoader):
    """
    Wrapper around UnstructuredFileLoader that captures errors during load.
    """
    def load(self) -> List[Document]:
        """Load file using UnstructuredFileLoader with error handling."""
        try:
            # Make sure 'unstructured' dependency is installed
            return super().load()
        except Exception as e:
            logger.error(f"Error loading file {self.file_path} with UnstructuredFileLoader: {str(e)}")
            # Return empty list on error, so DirectoryLoader can continue
            return []

class ConfigurableRAGPipeline:
    def __init__(
        self,
        llm_model: str = "llama3.1:latest",
        embedding_model: str = "nomic-embed-text:latest",
        temperature: float = 0.1,
        k_docs: int = 5,
        use_reranking: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_k: int = 3,
        # --- Chunking Configuration ---
        chunking_strategy: str = "sentence_window", # 'recursive', 'sentence_window', 'semantic'
        chunk_size: int = 1000, # For recursive
        chunk_overlap: int = 200, # For recursive
        sentence_chunk_overlap: int = 50, # For sentence_window's SentenceSplitter
        sentence_tokens_per_chunk: int = 256, # For sentence_window's SentenceSplitter
        semantic_threshold_type: str = "percentile", # For semantic: "percentile", "standard_deviation", "interquartile"
        semantic_threshold_amount: float = 0.95, # For semantic: e.g., 0.95 for 95th percentile
        # --- Data/Persistence ---
        data_dir: str = "data",
        persist_dir: str = "chroma_db"
    ):
        """
        Initialize the Configurable RAG pipeline.

        Args:
            llm_model (str): The Ollama LLM model identifier.
            embedding_model (str): The Ollama embedding model identifier.
            temperature (float): Temperature for LLM generation.
            k_docs (int): Number of documents to initially retrieve from the vector store.
            use_reranking (bool): Whether to use a cross-encoder reranker after retrieval.
            rerank_model (str): The HuggingFace model name for the cross-encoder reranker.
            rerank_top_k (int): Number of documents to keep after reranking.
            chunking_strategy (str): Method for splitting documents ('recursive', 'sentence_window', 'semantic').
            chunk_size (int): Chunk size for RecursiveCharacterTextSplitter.
            chunk_overlap (int): Chunk overlap for RecursiveCharacterTextSplitter.
            sentence_chunk_overlap (int): Chunk overlap for SentenceTransformersTokenTextSplitter (used in 'sentence_window').
            sentence_tokens_per_chunk (int): Tokens per chunk for SentenceTransformersTokenTextSplitter (used in 'sentence_window').
            semantic_threshold_type (str): Breakpoint threshold type for SemanticChunker.
            semantic_threshold_amount (float): Breakpoint threshold amount for SemanticChunker.
            data_dir (str): Directory containing source documents.
            persist_dir (str): Directory to persist the Chroma vector store.
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.k_docs = k_docs
        self.use_reranking = use_reranking
        self.rerank_model = rerank_model
        self.rerank_top_k = rerank_top_k

        # Chunking parameters
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_chunk_overlap = sentence_chunk_overlap
        self.sentence_tokens_per_chunk = sentence_tokens_per_chunk
        self.semantic_threshold_type = semantic_threshold_type
        self.semantic_threshold_amount = semantic_threshold_amount

        self.data_dir = data_dir
        self.persist_dir = persist_dir

        # Initialize components to None
        self.embeddings: Optional[OllamaEmbeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.llm: Optional[OllamaLLM] = None
        self.retriever: Optional[BaseRetriever] = None
        self.qa_chain: Optional[RetrievalQA] = None

        # Check if vectorstore exists
        self.vectorstore_ready = self._index_exists()
        logger.info(f"Vectorstore directory '{self.persist_dir}' exists: {self.vectorstore_ready}")

    def _index_exists(self) -> bool:
        """Check if a vector store index already exists and is usable."""
        return os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0

    def index_documents(self, force_reindex: bool = False):
        """
        Loads, chunks, embeds, and indexes documents from the data directory based on the chosen strategy.
        """
        logger.info(f"Starting document indexing from '{self.data_dir}'...")
        logger.info(f"Force reindex: {force_reindex}")
        logger.info(f"Chunking strategy: {self.chunking_strategy}")

        if self.vectorstore_ready and not force_reindex:
            logger.info(f"Index already exists at '{self.persist_dir}'. Skipping indexing.")
            return "Index already exists. Use --force-reindex to overwrite."

        # --- Path Validation and Debugging ---
        absolute_data_dir = os.path.abspath(self.data_dir)
        logger.info(f"Attempting to load documents from absolute path: {absolute_data_dir}")
        if not os.path.exists(absolute_data_dir):
            logger.error(f"Confirmed: Absolute data directory does not exist: {absolute_data_dir}")
            return f"Error: Data directory not found at {absolute_data_dir}"
        else:
            logger.info(f"Confirmed: Data directory exists at {absolute_data_dir}")
            try:
                dir_contents = os.listdir(absolute_data_dir)
                logger.info(f"Contents of data directory ({len(dir_contents)} items): {dir_contents[:10]}...") # Log first 10 items
            except Exception as e:
                 logger.error(f"Could not list contents of data directory: {e}")
        # --- End Path Validation ---


        # Remove existing index if force_reindex is True
        if force_reindex and os.path.exists(self.persist_dir):
            logger.warning(f"Force reindex requested. Removing existing index at '{self.persist_dir}'...")
            import shutil
            try:
                shutil.rmtree(self.persist_dir)
                logger.info("Existing index removed.")
            except OSError as e:
                logger.error(f"Error removing existing index: {e}")
                return f"Error removing existing index: {e}"

        # Initialize embeddings - needed for SemanticChunker and vector store creation
        if self.embeddings is None:
             logger.info(f"Initializing embedding model: {self.embedding_model}")
             # Add error handling for embedding model initialization
             try:
                self.embeddings = OllamaEmbeddings(model=self.embedding_model)
                logger.info(f"Embedding model '{self.embedding_model}' initialized.")
             except Exception as e:
                 logger.error(f"Failed to initialize embedding model '{self.embedding_model}': {e}", exc_info=True)
                 return f"Error initializing embedding model: {e}"


        # --- Document Loading ---
        loaded_documents: List[Document] = []
        try:
            logger.info(f"Scanning for documents in '{self.data_dir}'...")
            # Define specific loaders for known types
            # Ensure necessary packages are installed (e.g., pip install pypdf unstructured[pdf])
            loaders_map = {
                ".txt": TextLoader,
                ".pdf": PyPDFLoader,
                # Add more as needed, ensure SafeUnstructuredFileLoader handles others
                # ".docx": UnstructuredFileLoader, # Example
                # ".html": UnstructuredHTMLLoader, # Example
            }
            # Default loader for unknown types or those explicitly mapped to Unstructured
            default_loader_cls = SafeUnstructuredFileLoader

            p = Path(self.data_dir)
            all_files_paths = list(p.glob("**/*.*")) # Get all files first

            if not all_files_paths:
                 logger.warning(f"No files found matching '**/*.*' in data directory: {self.data_dir}")
                 return "No documents successfully loaded. Check logs for individual file errors."

            logger.info(f"Found {len(all_files_paths)} potential files. Attempting to load...")

            processed_files = set() # Keep track of files processed by specific loaders

            # Process files with specific loaders first
            for ext, loader_cls in loaders_map.items():
                 pattern = f"**/*{ext.lower()}" # Use lower case extension
                 ext_files = list(p.glob(pattern))
                 # Also check upper case if needed, depending on OS sensitivity
                 # ext_files.extend(list(p.glob(f"**/*{ext.upper()}")))

                 logger.info(f"Processing {len(ext_files)} files with {loader_cls.__name__} for extension '{ext}'")
                 for file_path in ext_files:
                     file_str = str(file_path)
                     if file_str in processed_files: continue # Skip if already processed

                     try:
                         logger.debug(f"Loading file: {file_str} using {loader_cls.__name__}")
                         if loader_cls == TextLoader:
                             loader_instance = loader_cls(file_str, encoding='utf-8', autodetect_encoding=True) # Try auto-detect
                         else:
                             loader_instance = loader_cls(file_str)

                         docs = loader_instance.load() # Load can return List[Document]
                         if isinstance(docs, list):
                            for doc in docs:
                                if isinstance(doc, Document):
                                    # Add source if missing (some loaders might not add it)
                                    if 'source' not in doc.metadata:
                                        doc.metadata['source'] = file_str
                                    loaded_documents.append(doc)
                         elif isinstance(docs, Document): # Should not happen with these loaders but check
                             if 'source' not in docs.metadata:
                                 docs.metadata['source'] = file_str
                             loaded_documents.append(docs)

                         processed_files.add(file_str)
                     except Exception as e:
                         logger.error(f"Failed to load {file_str} using {loader_cls.__name__}: {e}", exc_info=False)

            # Process remaining files with the default loader
            remaining_files = [f for f in all_files_paths if str(f) not in processed_files and not f.is_dir()]
            logger.info(f"Processing {len(remaining_files)} remaining files with {default_loader_cls.__name__}")
            for file_path in remaining_files:
                 file_str = str(file_path)
                 try:
                     logger.debug(f"Loading file: {file_str} using {default_loader_cls.__name__}")
                     loader_instance = default_loader_cls(file_str)
                     docs = loader_instance.load() # Safe loader returns List[Document] or []
                     for doc in docs: # Iterate even if empty
                         if isinstance(doc, Document):
                             if 'source' not in doc.metadata:
                                 doc.metadata['source'] = file_str
                             loaded_documents.append(doc)
                     processed_files.add(file_str) # Mark as processed even if load failed inside SafeLoader
                 except Exception as e:
                     # Should be caught by SafeLoader, but maybe instantiation fails
                     logger.error(f"Failed to instantiate or load {file_str} using {default_loader_cls.__name__}: {e}", exc_info=False)


            if not loaded_documents:
                logger.warning("No documents were successfully loaded after processing all found files.")
                return "No documents successfully loaded. Check logs for individual file errors."

            logger.info(f"Successfully loaded a total of {len(loaded_documents)} documents from {len(processed_files)} files.")

        except Exception as e:
            logger.error(f"Error during the document loading process: {e}", exc_info=True)
            return f"Error during document loading: {e}"
        # --- End Document Loading ---


        # --- Document Splitting ---
        chunks: List[Document] = []
        try:
            logger.info("Splitting documents into chunks...")
            if self.chunking_strategy == 'semantic':
                logger.info(f"Using SemanticChunker (threshold: {self.semantic_threshold_type} / {self.semantic_threshold_amount})")
                if self.embeddings is None: # Should have been initialized, but double check
                     raise ValueError("Embeddings must be initialized for SemanticChunker.")
                semantic_splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type=self.semantic_threshold_type,
                    breakpoint_threshold_amount=self.semantic_threshold_amount,
                )
                for i, doc in enumerate(loaded_documents):
                    logger.debug(f"Applying SemanticChunker to doc {i+1}/{len(loaded_documents)} (source: {doc.metadata.get('source', 'N/A')})")
                    # Handle potential errors during semantic chunking of a single doc
                    try:
                        doc_chunks = semantic_splitter.split_documents([doc])
                        for chunk in doc_chunks: chunk.metadata.update(doc.metadata)
                        chunks.extend(doc_chunks)
                    except Exception as sem_ex:
                        logger.error(f"Error applying SemanticChunker to {doc.metadata.get('source', 'N/A')}: {sem_ex}", exc_info=False)
                        # Optionally, fall back to recursive chunking for this doc
                        # rec_splitter = RecursiveCharacterTextSplitter(...)
                        # chunks.extend(rec_splitter.split_documents([doc]))
                logger.info(f"Created {len(chunks)} semantic chunks.")

            elif self.chunking_strategy == 'sentence_window':
                logger.info(f"Using sentence window chunking (Recursive then SentenceSplitter)")
                pre_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                pre_split_docs = pre_splitter.split_documents(loaded_documents)
                logger.info(f"Created {len(pre_split_docs)} initial recursive chunks.")
                sentence_splitter = SentenceTransformersTokenTextSplitter(
                    chunk_overlap=self.sentence_chunk_overlap, tokens_per_chunk=self.sentence_tokens_per_chunk
                )
                chunks = sentence_splitter.split_documents(pre_split_docs)
                logger.info(f"Created {len(chunks)} sentence-based chunks.")

            else: # Default to 'recursive'
                logger.info(f"Using RecursiveCharacterTextSplitter (size={self.chunk_size}, overlap={self.chunk_overlap})")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, # Use the instance attribute
                    chunk_overlap=self.chunk_overlap, # Use the instance attribute
                    length_function=len,
                    is_separator_regex=False, # Treat separators as literal strings
                    separators=[
                        # 1. Major Document Breaks (highest priority)
                        # Try to split between the main Senate Rules and appended documents
                        "\n\n---\n\n# RESOLUTION", # Separator before Resolutions
                        "\n\n---\n\n# The Constitution", # Separator before Constitution
                        "\n\n---\n\n# Ordinance", # Separator before Ordinance
                        "\n\n---\n\n", # General separator if specific titles aren't matched

                        # 2. Top-level Rules / Articles (often preceded by blank line & ##)
                        "\n\n## RULE ", # Split before Senate Rules (e.g., RULE I, RULE II)
                        "\n\n## ARTICLE ", # Split before Constitution Articles (e.g., ARTICLE I)

                        # 3. Section Breaks (various formats)
                        # Try to split before different section formats found in the docs
                        "\n### Section ", # Main Senate Rules sections (e.g., ### Section 1)
                        "\n## Sec. ",    # Resolution sections (e.g., ## Sec. 4.)
                        "\n## SEC. ",    # Constitution sections (e.g., ## SEC. 3.)
                        "\nSEC. ",       # Constitution sections if not preceded by ##
                        "\nSection ",    # Constitution/Ordinance sections (e.g., Section 1:)

                        # 4. Paragraphs (very common and important)
                        "\n\n", # Split by double newline (paragraph break)

                        # 5. Numbered/Lettered Lists Items (often start indented)
                        "\n  ", # Split before indented lines (common in lists like a), 1))

                        # 6. Line Breaks
                        "\n", # Split by single newline

                        # 7. Sentence Breaks (prefer splitting after punctuation + space)
                        ". ", # Split after period and space
                        "? ", # Split after question mark and space
                        "! ", # Split after exclamation mark and space

                        # 8. Clause / Phrase Breaks
                        "; ", # Split after semicolon and space
                        ": ", # Split after colon and space
                        ", ", # Split after comma and space

                        # 9. Parentheses / Brackets (might indicate distinct clauses/notes)
                        " (",
                        ") ",
                        "[",
                        "]",

                        # 10. Word Breaks
                        " ", # Split by space

                        # 11. Fallback Punctuation / Characters (if other splits fail)
                        ".",
                        ",",
                        ";",
                        ":",
                        "?",
                        "!",
                        "-",
                        "—", # Em dash
                        "", # Final fallback: split by character
                    ]
                )
                chunks = text_splitter.split_documents(loaded_documents)
                logger.info(f"Created {len(chunks)} recursive chunks.")

            # Add unique chunk IDs and ensure source information is properly captured
            for i, chunk in enumerate(chunks):
                # Get source information or use a fallback
                source = chunk.metadata.get('source', 'unknown_source')
                # Extract just the filename without the full path
                filename = os.path.basename(source)
                # Add chunk_id to metadata
                chunk.metadata["chunk_id"] = f"{filename}_chunk_{i}"
                # Ensure source is always set - this is critical for tracking
                if 'source' not in chunk.metadata or not chunk.metadata['source']:
                    chunk.metadata['source'] = filename

            if not chunks:
                logger.warning("No chunks were created from the loaded documents.")
                return "No chunks created. Check document content and chunking parameters."

        except Exception as e:
            logger.error(f"Error splitting documents using '{self.chunking_strategy}' strategy: {e}", exc_info=True)
            return f"Error splitting documents: {e}"
        # --- End Document Splitting ---

        # --- Vector Store Creation ---
        try:
            logger.info(f"Creating vector store at '{self.persist_dir}' with {len(chunks)} chunks...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks, embedding=self.embeddings, persist_directory=self.persist_dir
            )
            self.vectorstore_ready = True
            logger.info("Vector store created and persisted successfully.")
            return f"Successfully indexed {len(chunks)} chunks from {len(loaded_documents)} documents using '{self.chunking_strategy}'."

        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            if os.path.exists(self.persist_dir):
                logger.warning(f"Attempting to clean up failed vector store directory: {self.persist_dir}")
                import shutil
                try: shutil.rmtree(self.persist_dir)
                except OSError as rm_err: logger.error(f"Could not clean up directory: {rm_err}")
            return f"Error creating vector store: {e}"
        # --- End Vector Store Creation ---

    def initialize_components(self) -> bool:
        """
        Initializes the core components (embeddings, LLM, vector store, retriever, QA chain).
        """
        logger.info("Initializing RAG pipeline components...")
        try:
            # 1. Initialize Embeddings
            if self.embeddings is None:
                logger.info(f"Initializing embedding model: {self.embedding_model}")
                self.embeddings = OllamaEmbeddings(model=self.embedding_model)

            # 2. Load Vector Store
            if not self.vectorstore_ready:
                logger.error("Vectorstore is not ready. Please run index_documents() first.")
                return False
            if self.vectorstore is None:
                logger.info(f"Loading vector store from: {self.persist_dir}")
                self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
                # Verify connection after loading
                try:
                    logger.info(f"Vector store loaded. Collection count: {self.vectorstore._collection.count()}")
                except Exception as vs_err:
                     logger.error(f"Failed to verify vector store connection after loading: {vs_err}", exc_info=True)
                     return False
                logger.info("Vector store loaded successfully.")


            # 3. Initialize LLM
            if self.llm is None:
                logger.info(f"Initializing LLM: {self.llm_model} with temperature {self.temperature}")
                self.llm = OllamaLLM(model=self.llm_model, temperature=self.temperature)
                # Test LLM connection (optional but recommended)
                try:
                    logger.info("Testing LLM connection...")
                    self.llm.invoke("Respond with 'OK'")
                    logger.info("LLM connection successful.")
                except Exception as llm_err:
                    logger.error(f"Failed to connect to LLM '{self.llm_model}': {llm_err}", exc_info=False)
                    return False

            # 4. Initialize Retriever
            logger.info("Initializing retriever...")
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_docs})
            logger.info(f"Base retriever configured to retrieve top {self.k_docs} documents.")

            if self.use_reranking:
                logger.info(f"Reranking enabled. Using cross-encoder: {self.rerank_model}")
                try:
                    cross_encoder = HuggingFaceCrossEncoder(model_name=self.rerank_model)
                    reranker = CrossEncoderReranker(model=cross_encoder, top_n=self.rerank_top_k)
                    self.retriever = ContextualCompressionRetriever(
                        base_compressor=reranker, base_retriever=base_retriever
                    )
                    logger.info(f"Contextual Compression Retriever with reranking configured (top_n={self.rerank_top_k}).")
                except Exception as e:
                    logger.error(f"Failed to initialize reranker model '{self.rerank_model}'. Falling back to base retriever. Error: {e}", exc_info=True)
                    self.retriever = base_retriever
                    self.use_reranking = False
            else:
                logger.info("Reranking disabled. Using base vector store retriever.")
                self.retriever = base_retriever

            # 5. Initialize QA Chain
            logger.info("Setting up RetrievalQA chain...")
            self._setup_standard_qa_chain()
            logger.info("RetrievalQA chain configured.")

            logger.info("All RAG components initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}", exc_info=True)
            return False

    def _setup_standard_qa_chain(self):
        """Sets up the standard RetrievalQA chain."""
        template = """
        You are a helpful AI assistant. Answer the following question based ONLY on the context provided.
        If the context does not contain the information needed to answer the question, state explicitly "I cannot answer the question based on the provided context."
        Do not make up information or answer from your general knowledge. Be concise and directly address the question.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}, return_source_documents=True
        )
        logger.info("Standard RetrievalQA chain created.")

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Answers a given query using the configured RAG pipeline."""
        logger.info(f"Received query: {query}")
        if self.qa_chain is None:
            logger.error("QA chain is not initialized. Call initialize_components() first.")
            return {"answer": "Error: Pipeline not initialized.", "sources": []}

        try:
            logger.info("Invoking QA chain...")
            result = self.qa_chain.invoke({"query": query})
            answer = result.get("result", "Error: Could not generate answer.")
            source_documents = result.get("source_documents", [])
            logger.info(f"Generated answer length: {len(answer)} characters.")
            logger.info(f"Retrieved {len(source_documents)} source documents after processing.")

            sources_metadata = []
            for idx, doc in enumerate(source_documents):
                meta = doc.metadata.copy() if hasattr(doc, "metadata") else {}
                
                # Ensure source has a valid value and is a string
                source = meta.get("source", "Unknown")
                # Log each source for debugging
                logger.info(f"Source document {idx+1}: {source}")
                
                # Create simplified metadata that's easy to parse by evaluation script
                simplified_meta = {
                    "source": source,
                    "chunk_id": meta.get("chunk_id", f"unknown_chunk_{idx}"),
                    # Check for specific score key from reranker
                    "score": meta.get("relevance_score", meta.get("score", None)),
                    "content_snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                }
                sources_metadata.append(simplified_meta)

            return {"answer": answer, "sources": sources_metadata}
        except Exception as e:
            logger.error(f"Error during QA chain invocation: {e}", exc_info=True)
            return {"answer": f"Error processing query: {e}", "sources": []}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a configurable RAG pipeline without UI")

    # Core Configuration
    parser.add_argument("--llm", type=str, default="llama3.1:latest", help="Ollama LLM model identifier")
    parser.add_argument("--embeddings", type=str, default="bge-m3", help="Ollama embedding model identifier")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM generation")

    # Retrieval Configuration
    parser.add_argument("--k-docs", type=int, default=10, help="Number of documents to initially retrieve")
    parser.add_argument("--no-reranking", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--rerank-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="HuggingFace model for reranker") # Changed default
    parser.add_argument("--rerank-top-k", type=int, default=5, help="Number of documents to keep after reranking")

    # Indexing Configuration
    parser.add_argument("--chunking-strategy", type=str, default="sentence_window",
                        choices=["recursive", "sentence_window", "semantic"], help="Document chunking method")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size for 'recursive' & 'sentence_window' (initial split)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for 'recursive' & 'sentence_window' (initial split)")
    parser.add_argument("--sentence-chunk-overlap", type=int, default=50, help="Overlap for 'sentence_window' strategy's SentenceSplitter")
    parser.add_argument("--sentence-tokens-per-chunk", type=int, default=256, help="Tokens per chunk for 'sentence_window' strategy's SentenceSplitter")
    parser.add_argument("--semantic-threshold-type", type=str, default="percentile",
                        choices=["percentile", "standard_deviation", "interquartile"], help="Threshold type for 'semantic' strategy")
    parser.add_argument("--semantic-threshold-amount", type=float, default=0.75, help="Threshold amount for 'semantic' strategy (e.g., 0.95 for 95th percentile). Tune this value.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing source documents")
    parser.add_argument("--persist-dir", type=str, default="chroma_db", help="Directory to persist the vector store")
    parser.add_argument("--force-reindex", action="store_true", help="Force deletion and reindexing of documents")

    # Execution Mode
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("query", nargs="?", default=None, help="Single query to answer (if not interactive)")

    args = parser.parse_args()

    # --- Initialize Pipeline ---
    print("--- Initializing Configurable RAG Pipeline ---")
    pipeline = ConfigurableRAGPipeline(
        llm_model=args.llm,
        embedding_model=args.embeddings,
        temperature=args.temperature,
        k_docs=args.k_docs,
        use_reranking=not args.no_reranking,
        rerank_model=args.rerank_model,
        rerank_top_k=args.rerank_top_k,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sentence_chunk_overlap=args.sentence_chunk_overlap,
        sentence_tokens_per_chunk=args.sentence_tokens_per_chunk,
        semantic_threshold_type=args.semantic_threshold_type,
        semantic_threshold_amount=args.semantic_threshold_amount,
        data_dir=args.data_dir,
        persist_dir=args.persist_dir
    )
    print("Pipeline Class Instantiated.")
    print("\n--- Configuration ---")
    print(f"LLM Model: {pipeline.llm_model}")
    print(f"Embedding Model: {pipeline.embedding_model}")
    print(f"Temperature: {pipeline.temperature}")
    print(f"Retrieve K Docs: {pipeline.k_docs}")
    print(f"Use Reranking: {pipeline.use_reranking}")
    if pipeline.use_reranking:
        print(f"  Reranker Model: {pipeline.rerank_model}")
        print(f"  Rerank Top K: {pipeline.rerank_top_k}")
    print(f"Chunking Strategy: {pipeline.chunking_strategy}")
    if pipeline.chunking_strategy == 'semantic':
        print(f"  Semantic Threshold Type: {pipeline.semantic_threshold_type}")
        print(f"  Semantic Threshold Amount: {pipeline.semantic_threshold_amount}")
    elif pipeline.chunking_strategy == 'sentence_window':
         print(f"  Sentence Splitter Overlap: {pipeline.sentence_chunk_overlap}")
         print(f"  Sentence Splitter Tokens/Chunk: {pipeline.sentence_tokens_per_chunk}")
         print(f"  Recursive Chunk Size (Initial): {pipeline.chunk_size}")
         print(f"  Recursive Chunk Overlap (Initial): {pipeline.chunk_overlap}")
    else: # recursive
        print(f"  Recursive Chunk Size: {pipeline.chunk_size}")
        print(f"  Recursive Chunk Overlap: {pipeline.chunk_overlap}")
    print(f"Data Directory: {pipeline.data_dir}")
    print(f"Persist Directory: {pipeline.persist_dir}")
    print("---------------------\n")

    # --- Indexing ---
    print("--- Document Indexing Check ---")
    indexing_result = pipeline.index_documents(force_reindex=args.force_reindex)
    print(f"Indexing Result: {indexing_result}")
    if "Error" in indexing_result and not pipeline._index_exists(): # Check if index really doesn't exist after error
        print("Exiting due to indexing error and no usable index found.")
        return
    # Update vectorstore_ready state after attempting indexing
    pipeline.vectorstore_ready = pipeline._index_exists()
    print("-----------------------------\n")

    # --- Initialize Components ---
    print("--- Initializing Pipeline Components ---")
    if not pipeline.initialize_components():
        print("Failed to initialize pipeline components. Exiting.")
        return
    print("Pipeline components initialized successfully.")
    print("--------------------------------------\n")

    print("--- RAG Pipeline Ready ---")

    # --- Execution Modes ---
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end.")
        while True:
            try:
                query = input("\nAsk a question: ")
                if query.lower() in ["exit", "quit", "q"]: break
                if not query.strip(): continue
                print("Processing...")
                result = pipeline.answer_question(query)
                print("\nAnswer:")
                print(result["answer"])
                print("\nSources:")
                if result["sources"]:
                    for i, source_meta in enumerate(result["sources"]):
                        print(f"  Source {i+1}:")
                        print(f"    File: {source_meta.get('source', 'N/A')}")
                        print(f"    Chunk ID: {source_meta.get('chunk_id', 'N/A')}")
                        score = source_meta.get('score')
                        if score is not None: print(f"    Score: {score:.4f}")
                        # print(f"    Snippet: {source_meta.get('content_snippet', 'N/A')}")
                else:
                    print("  No source documents returned.")
            except EOFError: break
            except KeyboardInterrupt: break
            except Exception as e:
                 logger.error(f"Error in interactive loop: {e}", exc_info=True)
                 print(f"An error occurred: {e}")
        print("\nExiting interactive mode.")

    elif args.query:
        print(f"\n--- Answering Single Query ---")
        print(f"Query: {args.query}")
        print("Processing...")
        result = pipeline.answer_question(args.query)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        if result["sources"]:
            for i, source_meta in enumerate(result["sources"]):
                print(f"  Source {i+1}:")
                print(f"    File: {source_meta.get('source', 'N/A')}")
                print(f"    Chunk ID: {source_meta.get('chunk_id', 'N/A')}")
                score = source_meta.get('score')
                if score is not None: print(f"    Score: {score:.4f}")
                # print(f"    Snippet: {source_meta.get('content_snippet', 'N/A')}")
        else:
            print("  No source documents returned.")
        print("----------------------------\n")

    else:
        print("\nNo query provided and not in interactive mode.")
        print("Use 'python configurable_rag.py --interactive' or 'python configurable_rag.py \"Your question?\"'")

if __name__ == "__main__":
    main()
