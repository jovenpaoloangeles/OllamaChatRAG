import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_ollama import OllamaEmbeddings
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def index_exists(persist_dir="chroma_db"):
    """Check if a vector store index already exists"""
    return os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0

# Custom loader class with better error handling
class SafeUnstructuredFileLoader(UnstructuredFileLoader):
    def load(self):
        try:
            return super().load()
        except Exception as e:
            logger.error(f"Error loading file {self.file_path}: {str(e)}")
            return []

def create_vectorstore(data_dir="data", persist_dir="chroma_db", chunk_size=1000, chunk_overlap=200, force_reindex=False, use_sentence_window=True):
    """Create a vector store from documents in the data directory"""
    # Check if index already exists and force_reindex is False
    if index_exists(persist_dir) and not force_reindex:
        return "Index already exists. Use force_reindex=True to recreate the index."
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist. Creating it...")
        os.makedirs(data_dir)
        return "No documents found. Please add documents to the data directory."
    
    # Load documents
    try:
        logger.info(f"Starting to load documents from {data_dir}")
        
        # Get all PDF files recursively
        pdf_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Load each file individually with error handling
        documents = []
        for file_path in pdf_files:
            try:
                logger.info(f"Loading {file_path}")
                loader = SafeUnstructuredFileLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        if not documents:
            logger.warning("No documents were successfully loaded")
            return "No documents were successfully loaded. Check the logs for errors."
        
        logger.info(f"Total documents loaded: {len(documents)}")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks")
        
        if use_sentence_window:
            # First split into smaller chunks using RecursiveCharacterTextSplitter
            pre_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            pre_split_docs = pre_splitter.split_documents(documents)
            logger.info(f"Created {len(pre_split_docs)} initial chunks from {len(documents)} documents")
            
            # Then apply sentence transformer token splitter for better semantic chunking
            sentence_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=50,
                tokens_per_chunk=256
            )
            chunks = sentence_splitter.split_documents(pre_split_docs)
            logger.info(f"Created {len(chunks)} sentence-based chunks from {len(pre_split_docs)} initial chunks")
            
            # Add source document metadata for better traceability
            for i, chunk in enumerate(chunks):
                if "source" in chunk.metadata:
                    chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_chunk_{i}"
                else:
                    chunk.metadata["chunk_id"] = f"chunk_{i}"
        else:
            # Use traditional chunking method
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create embeddings using Ollama's nomic-embed-text
        logger.info("Creating embeddings using nomic-embed-text")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Remove existing vector store if it exists
        if os.path.exists(persist_dir):
            logger.info(f"Removing existing vector store at {persist_dir}")
            shutil.rmtree(persist_dir)
        
        # Create vector store
        logger.info("Creating vector store")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        # No need to call persist() as it's automatically persisted in newer versions
        logger.info("Vector store created successfully")
        
        return f"Successfully indexed {len(chunks)} chunks from {len(documents)} documents"
    
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return f"Error creating vector store: {str(e)}"

if __name__ == "__main__":
    result = create_vectorstore()
    print(result)
