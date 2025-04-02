# simple_rag.py
import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.schema.retriever import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SimpleRAG:
    def __init__(
        self,
        llm_model: str = "llama3.1:latest",
        embedding_model: str = "nomic-embed-text:latest",
        temperature: float = 0.1,
        k_docs: int = 5,
        use_reranking: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_k: int = 3,
        persist_dir: str = "chroma_db"
    ):
        """
        Initialize the Simple RAG pipeline that uses an existing vector store.

        Args:
            llm_model (str): The Ollama LLM model identifier.
            embedding_model (str): The Ollama embedding model identifier.
            temperature (float): Temperature for LLM generation.
            k_docs (int): Number of documents to initially retrieve from the vector store.
            use_reranking (bool): Whether to use a cross-encoder reranker after retrieval.
            rerank_model (str): The HuggingFace model name for the cross-encoder reranker.
            rerank_top_k (int): Number of documents to keep after reranking.
            persist_dir (str): Directory where the Chroma vector store is persisted.
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.k_docs = k_docs
        self.use_reranking = use_reranking
        self.rerank_model = rerank_model
        self.rerank_top_k = rerank_top_k
        self.persist_dir = persist_dir

        # Initialize components to None
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.qa_chain = None

        # Check if vectorstore exists
        self.vectorstore_ready = self._index_exists()
        logger.info(f"Vectorstore directory '{self.persist_dir}' exists: {self.vectorstore_ready}")

    def _index_exists(self) -> bool:
        """Check if a vector store index already exists and is usable."""
        return os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0

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
                logger.error("Vectorstore is not ready. Please create an index first using document_chunker.py")
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
            self._setup_qa_chain()
            logger.info("RetrievalQA chain configured.")

            logger.info("All RAG components initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}", exc_info=True)
            return False

    def _setup_qa_chain(self):
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
                
                # Create simplified metadata that's easy to parse
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
    parser = argparse.ArgumentParser(description="Run a simple RAG pipeline using an existing vector store")

    # Core Configuration
    parser.add_argument("--llm", type=str, default="llama3.1:latest", help="Ollama LLM model identifier")
    parser.add_argument("--embeddings", type=str, default="nomic-embed-text:latest", help="Ollama embedding model identifier")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for LLM generation")

    # Retrieval Configuration
    parser.add_argument("--k-docs", type=int, default=10, help="Number of documents to initially retrieve")
    parser.add_argument("--no-reranking", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--rerank-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="HuggingFace model for reranker")
    parser.add_argument("--rerank-top-k", type=int, default=5, help="Number of documents to keep after reranking")

    # Vector store location
    parser.add_argument("--persist-dir", type=str, default="chroma_db", help="Directory where the vector store is persisted")

    # Execution Mode
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("query", nargs="?", default=None, help="Single query to answer (if not interactive)")

    args = parser.parse_args()

    # --- Initialize Pipeline ---
    print("--- Initializing Simple RAG Pipeline ---")
    rag = SimpleRAG(
        llm_model=args.llm,
        embedding_model=args.embeddings,
        temperature=args.temperature,
        k_docs=args.k_docs,
        use_reranking=not args.no_reranking,
        rerank_model=args.rerank_model,
        rerank_top_k=args.rerank_top_k,
        persist_dir=args.persist_dir
    )
    print("Pipeline Class Instantiated.")
    print("\n--- Configuration ---")
    print(f"LLM Model: {rag.llm_model}")
    print(f"Embedding Model: {rag.embedding_model}")
    print(f"Temperature: {rag.temperature}")
    print(f"Retrieve K Docs: {rag.k_docs}")
    print(f"Use Reranking: {rag.use_reranking}")
    if rag.use_reranking:
        print(f"  Reranker Model: {rag.rerank_model}")
        print(f"  Rerank Top K: {rag.rerank_top_k}")
    print(f"Persist Directory: {rag.persist_dir}")
    print("---------------------\n")

    # --- Check Vector Store ---
    if not rag.vectorstore_ready:
        print("Error: Vector store not found or empty. Please create an index first using document_chunker.py")
        return

    # --- Initialize Components ---
    print("--- Initializing Pipeline Components ---")
    if not rag.initialize_components():
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
                result = rag.answer_question(query)
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
        result = rag.answer_question(args.query)
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
        else:
            print("  No source documents returned.")
        print("----------------------------\n")

    else:
        print("\nNo query provided and not in interactive mode.")
        print("Use 'python simple_rag.py --interactive' or 'python simple_rag.py \"Your question?\"'")

if __name__ == "__main__":
    main()
