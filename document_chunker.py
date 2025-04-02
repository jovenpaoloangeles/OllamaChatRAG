# document_chunker.py
import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

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

class DocumentChunker:
    def __init__(
        self,
        embedding_model: str = "nomic-embed-text:latest",
        persist_dir: str = "chroma_db"
    ):
        """
        Initialize the Document Chunker.

        Args:
            embedding_model (str): The Ollama embedding model identifier.
            persist_dir (str): Directory to persist the Chroma vector store.
        """
        self.embedding_model = embedding_model
        self.persist_dir = persist_dir
        self.embeddings = None
        self.vectorstore = None
        self.vectorstore_ready = self._index_exists()

    def _index_exists(self) -> bool:
        """Check if a vector store index already exists and is usable."""
        return os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0

    def _initialize_embeddings(self):
        """Initialize the embedding model if not already initialized."""
        if self.embeddings is None:
            logger.info(f"Initializing embedding model: {self.embedding_model}")
            try:
                self.embeddings = OllamaEmbeddings(model=self.embedding_model)
                logger.info(f"Embedding model '{self.embedding_model}' initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model '{self.embedding_model}': {e}", exc_info=True)
                raise

    def _load_vectorstore(self):
        """Load the existing vector store if it exists."""
        if self.vectorstore is None and self.vectorstore_ready:
            logger.info(f"Loading vector store from: {self.persist_dir}")
            self._initialize_embeddings()
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            try:
                logger.info(f"Vector store loaded. Collection count: {self.vectorstore._collection.count()}")
            except Exception as vs_err:
                logger.error(f"Failed to verify vector store connection after loading: {vs_err}", exc_info=True)
                raise

    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document file and return a list of Document objects."""
        logger.info(f"Loading document: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Generate a document ID based on file path and name
        doc_id = os.path.normpath(file_path).replace("\\", "_").replace("/", "_")
        logger.info(f"Generated document ID: {doc_id}")
        
        # Define specific loaders for known types
        loaders_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
        }
        
        # Use appropriate loader based on file extension
        if file_ext in loaders_map:
            loader_cls = loaders_map[file_ext]
            try:
                if loader_cls == TextLoader:
                    loader_instance = loader_cls(file_path, encoding='utf-8', autodetect_encoding=True)
                else:
                    loader_instance = loader_cls(file_path)
                    
                docs = loader_instance.load()
                # Add document ID to metadata
                for doc in docs:
                    if isinstance(doc, Document):
                        # Add source if missing
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = file_path
                        # Add document ID
                        doc.metadata['doc_id'] = doc_id
                
                logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
                return docs
            except Exception as e:
                logger.error(f"Failed to load {file_path} using {loader_cls.__name__}: {e}")
                return []
        else:
            # Use default loader for unknown types
            try:
                loader = SafeUnstructuredFileLoader(file_path)
                docs = loader.load()
                # Add document ID to metadata
                for doc in docs:
                    if isinstance(doc, Document):
                        # Add source if missing
                        if 'source' not in doc.metadata:
                            doc.metadata['source'] = file_path
                        # Add document ID
                        doc.metadata['doc_id'] = doc_id
                
                logger.info(f"Successfully loaded {len(docs)} documents from {file_path} using SafeUnstructuredFileLoader")
                return docs
            except Exception as e:
                logger.error(f"Failed to load {file_path} using SafeUnstructuredFileLoader: {e}")
                return []

    def chunk_document(
        self, 
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """Split documents into chunks using recursive chunking."""
        logger.info(f"Chunking documents using recursive strategy")
        chunks = []
        
        try:
            logger.info(f"Using RecursiveCharacterTextSplitter (size={chunk_size}, overlap={chunk_overlap})")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=[
                    "\n\n---\n\n",  # Major document breaks
                    "\n\n## ",    # Top-level headers
                    "\n### ",    # Sub-headers
                    "\n\n",      # Paragraphs
                    "\n",        # Line breaks
                    ". ",        # Sentences
                    "? ",        # Questions
                    "! ",        # Exclamations
                    ", ",        # Clauses
                    " ",         # Words
                    ""           # Characters
                ]
            )
            chunks = text_splitter.split_documents(documents)
            
            # Ensure document ID is preserved in all chunks
            for i, chunk in enumerate(chunks):
                # Add chunk index to metadata to make each chunk uniquely identifiable
                chunk.metadata['chunk_index'] = i
                # Ensure doc_id is preserved from parent document
                if 'doc_id' not in chunk.metadata and len(documents) > 0:
                    # If there's only one source document, copy its doc_id
                    if 'doc_id' in documents[0].metadata:
                        chunk.metadata['doc_id'] = documents[0].metadata['doc_id']
            
            logger.info(f"Created {len(chunks)} recursive chunks.")

            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}", exc_info=True)
            raise

    def _get_existing_doc_ids(self) -> List[str]:
        """Get a list of document IDs that already exist in the vector store."""
        if not self.vectorstore_ready:
            return []
            
        if self.vectorstore is None:
            self._load_vectorstore()
            
        try:
            # Get all documents from the collection
            results = self.vectorstore.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            # Extract unique document IDs
            doc_ids = set()
            for metadata in metadatas:
                if metadata and "doc_id" in metadata:
                    doc_ids.add(metadata["doc_id"])
                    
            logger.info(f"Found {len(doc_ids)} existing document IDs in the vector store")
            return list(doc_ids)
        except Exception as e:
            logger.error(f"Error getting existing document IDs: {e}", exc_info=True)
            return []

    def _delete_documents_by_id(self, doc_id: str) -> bool:
        """Delete all documents with the given document ID from the vector store."""
        if not self.vectorstore_ready:
            return False
            
        if self.vectorstore is None:
            self._load_vectorstore()
            
        try:
            # Get all documents with the given doc_id
            where_filter = {"doc_id": doc_id}
            self.vectorstore.delete(where=where_filter)
            logger.info(f"Deleted documents with ID: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents with ID {doc_id}: {e}", exc_info=True)
            return False

    def add_to_index(self, chunks: List[Document], replace_existing: bool = True) -> bool:
        """Add document chunks to the vector store index.
        
        Args:
            chunks: List of document chunks to add
            replace_existing: If True, replace existing documents with the same ID
        """
        if not chunks:
            logger.warning("No chunks provided to add to index.")
            return False
            
        try:
            self._initialize_embeddings()
            
            # Create new vectorstore if it doesn't exist
            if not self.vectorstore_ready:
                logger.info(f"Creating new vector store at '{self.persist_dir}' with {len(chunks)} chunks...")
                self.vectorstore = Chroma.from_documents(
                    documents=chunks, embedding=self.embeddings, persist_directory=self.persist_dir
                )
                self.vectorstore_ready = True
                logger.info("Vector store created and persisted successfully.")
                return True
            
            # Load existing vectorstore if not already loaded
            if self.vectorstore is None:
                self._load_vectorstore()
            
            # Check for existing documents with the same ID
            if len(chunks) > 0 and "doc_id" in chunks[0].metadata:
                doc_id = chunks[0].metadata["doc_id"]
                existing_doc_ids = self._get_existing_doc_ids()
                
                if doc_id in existing_doc_ids:
                    if replace_existing:
                        logger.info(f"Document with ID '{doc_id}' already exists. Replacing...")
                        if not self._delete_documents_by_id(doc_id):
                            logger.error(f"Failed to delete existing document with ID '{doc_id}'")
                            return False
                    else:
                        logger.warning(f"Document with ID '{doc_id}' already exists. Skipping addition. Use --replace to overwrite.")
                        return False
            
            # Add documents to existing vectorstore
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            self.vectorstore.add_documents(chunks)
            # Note: Chroma automatically persists changes to disk, no need to call persist()
            logger.info("Chunks added to vector store successfully.")
            
            return True
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}", exc_info=True)
            return False

    def process_single_file(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        replace_existing: bool = True
    ) -> bool:
        """Process a single file: load, chunk, and add to index."""
        # Load the document
        documents = self.load_document(file_path)
        if not documents:
            logger.error(f"Failed to load any documents from {file_path}")
            return False
            
        # Chunk the document
        chunks = self.chunk_document(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        if not chunks:
            logger.error(f"Failed to create any chunks from {file_path}")
            return False
            
        # Add chunks to index
        return self.add_to_index(chunks, replace_existing=replace_existing)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Add documents to a vector store index with recursive chunking")

    # File or directory to process
    parser.add_argument("--file", type=str, help="Path to a single file to process")
    parser.add_argument("--dir", type=str, help="Path to a directory of files to process")
    
    # Embedding model
    parser.add_argument("--embeddings", type=str, default="nomic-embed-text:latest", help="Ollama embedding model identifier")
    
    # Vector store location
    parser.add_argument("--persist-dir", type=str, default="chroma_db", help="Directory to persist the vector store")
    
    # Chunking parameters
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for recursive chunking")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for recursive chunking")
    
    # Document handling
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip documents that already exist in the index instead of replacing them")

    args = parser.parse_args()

    # Validate input arguments
    if not args.file and not args.dir:
        print("Error: Either --file or --dir must be specified.")
        parser.print_help()
        return

    # Initialize the document chunker
    chunker = DocumentChunker(
        embedding_model=args.embeddings,
        persist_dir=args.persist_dir
    )

    # Process a single file
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
            
        print(f"Processing file: {args.file}")
        success = chunker.process_single_file(
            file_path=args.file,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            replace_existing=not args.skip_duplicates
        )
        if success:
            print(f"Successfully processed and indexed {args.file}")
        else:
            print(f"Failed to process {args.file}")

    # Process a directory of files
    if args.dir:
        if not os.path.exists(args.dir) or not os.path.isdir(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            return
            
        print(f"Processing directory: {args.dir}")
        
        # Find all files in the directory
        p = Path(args.dir)
        all_files = list(p.glob("**/*.*"))
        
        if not all_files:
            print(f"No files found in directory: {args.dir}")
            return
            
        print(f"Found {len(all_files)} files to process")
        
        # Process each file
        success_count = 0
        for file_path in all_files:
            if file_path.is_file():
                print(f"Processing file: {file_path}")
                success = chunker.process_single_file(
                    file_path=str(file_path),
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    replace_existing=not args.skip_duplicates
                )
                if success:
                    success_count += 1
                    print(f"Successfully processed and indexed {file_path}")
                else:
                    print(f"Failed to process {file_path}")
        
        print(f"Successfully processed {success_count} out of {len(all_files)} files")

if __name__ == "__main__":
    main()
