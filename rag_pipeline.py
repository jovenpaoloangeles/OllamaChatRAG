import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import networkx as nx
from collections import defaultdict
import numpy as np

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# GraphRAG
def graph_retrieval(vectorstore, query, k=100, start_k=30, adjacent_k=20, max_depth=5):
    """Retrieve documents using graph-based retrieval.
    
    Args:
        vectorstore: The vectorstore to use for retrieval
        query: The query to retrieve documents for
        k: Total number of nodes to retrieve
        start_k: Number of initial nodes to retrieve
        adjacent_k: Number of adjacent nodes to retrieve per node
        max_depth: Maximum depth of graph traversal
        
    Returns:
        List of retrieved documents
    """
    # Initial retrieval
    initial_docs = vectorstore.similarity_search(query, k=start_k)
    
    # Create a graph
    G = nx.Graph()
    
    # Add initial nodes
    for i, doc in enumerate(initial_docs):
        G.add_node(i, document=doc)
    
    # Traverse the graph up to max_depth
    frontier = list(range(len(initial_docs)))
    visited = set(frontier)
    
    for depth in range(max_depth):
        if not frontier:
            break
            
        next_frontier = []
        
        for node_id in frontier:
            node_doc = G.nodes[node_id]["document"]
            
            # Get similar documents to this node
            similar_docs = vectorstore.similarity_search(
                node_doc.page_content, k=adjacent_k
            )
            
            # Add edges to similar documents
            for j, similar_doc in enumerate(similar_docs):
                # Create a new node ID
                new_id = len(G.nodes)
                
                # Check if this document is already in the graph
                is_duplicate = False
                for existing_id in G.nodes:
                    if existing_id in visited and G.nodes[existing_id]["document"].page_content == similar_doc.page_content:
                        is_duplicate = True
                        new_id = existing_id
                        break
                
                if not is_duplicate:
                    G.add_node(new_id, document=similar_doc)
                    visited.add(new_id)
                    next_frontier.append(new_id)
                
                # Add edge between current node and similar document
                G.add_edge(node_id, new_id)
                
                # Stop if we've reached the total limit
                if len(G.nodes) >= k:
                    break
            
            if len(G.nodes) >= k:
                break
        
        frontier = next_frontier
        if len(G.nodes) >= k:
            break
    
    # Extract documents from the graph
    retrieved_docs = [G.nodes[i]["document"] for i in G.nodes]
    
    # Limit to k documents
    return retrieved_docs[:k]

# Simple implementation of lazy_graph_rag
def process_lazy_graph_rag(query, vectorstore, llm, prompt, fallback_retriever, graph_k=100, graph_start_k=30, graph_adjacent_k=20, graph_max_depth=3, rerank_top_k=5):
    """Process a query using LazyGraphRAG.
    
    Args:
        query: The query to process
        vectorstore: The vectorstore to use for retrieval
        llm: The language model to use
        prompt: The prompt template to use
        fallback_retriever: The fallback retriever to use if LazyGraphRAG fails
        graph_k: Total number of nodes to retrieve in graph traversal
        graph_start_k: Number of initial nodes to retrieve in graph traversal
        graph_adjacent_k: Number of adjacent nodes to retrieve per node
        graph_max_depth: Maximum depth of graph traversal
        rerank_top_k: Number of documents to keep after reranking
        
    Returns:
        Dict with answer, claims, and sources
    """
    try:
        # First, perform vector retrieval
        print(f"Performing vector retrieval for query: {query}")
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": graph_start_k})
        vector_docs = vector_retriever.get_relevant_documents(query)
        
        # Then, perform graph traversal
        print(f"Performing graph retrieval for query: {query}")
        graph_retriever = GraphRetriever(
            vectorstore=vectorstore,
            k=graph_k,
            start_k=graph_start_k,
            adjacent_k=graph_adjacent_k,
            max_depth=graph_max_depth
        )
        graph_docs = graph_retriever.get_relevant_documents(query, vector_docs)
        
        # Print content of graph_docs for analysis
        print(f"\nRetrieved {len(graph_docs)} documents from graph traversal:")
        for i, doc in enumerate(graph_docs[:3]):  # Show first 3 docs
            print(f"Document {i+1}/{len(graph_docs)}:")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"Content: {content_preview}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        if len(graph_docs) > 3:
            print(f"... and {len(graph_docs) - 3} more documents")
        
        # Extract claims from graph documents
        print("Extracting claims from graph documents...")
        claims = extract_claims_from_docs(graph_docs, query, llm)
        
        # Rerank claims
        print("Combining and reranking documents...")
        reranked_claims = rerank_claims(claims, query, rerank_top_k)
        print(f"Reranked to top {len(reranked_claims)} documents")
        
        # Format claims for prompt
        claims_text = "\n\n".join([f"{i+1}. {claim.page_content}" for i, claim in enumerate(reranked_claims)])
        
        # Get sources
        sources = []
        for claim in reranked_claims:
            if hasattr(claim, "metadata") and "source" in claim.metadata:
                sources.append(claim.metadata["source"])
        
        return {
            "claims": claims_text,
            "question": query,
            "sources": sources
        }
        
    except Exception as e:
        print(f"Error in LazyGraphRAG: {str(e)}")
        # Fall back to standard retrieval
        if fallback_retriever:
            print("Falling back to standard retrieval")
            docs = fallback_retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get sources
            sources = []
            for doc in docs:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.append(doc.metadata["source"])
            
            return {
                "context": context,
                "question": query,
                "sources": sources
            }
        else:
            return {
                "claims": "No claims found.",
                "question": query,
                "sources": []
            }

def extract_claims_from_docs(docs, query, llm):
    """Extract claims from documents using an LLM.
    
    Args:
        docs: List of documents to extract claims from
        query: The query to extract claims for
        llm: The language model to use
        
    Returns:
        List of Document objects containing claims
    """
    if not docs:
        return []
    
    print(f"Extracting claims from {len(docs)} documents...")
    
    # Create a prompt for claim extraction
    prompt = ChatPromptTemplate.from_template("""
    Extract factual claims from the following text that are relevant to answering the question.
    Each claim should be a single, atomic fact that helps answer the question.
    Return each claim on a new line prefixed with 'CLAIM: '.
    
    Question: {question}
    
    Text: {text}
    
    Relevant Claims:
    """)
    
    all_claims = []
    
    # Process each document
    for i, doc in enumerate(docs):
        try:
            print(f"Processing document {i+1}/{len(docs)}...")
            print(f"Document content: {doc.page_content[:200]}...")
            
            # Generate claims using the LLM
            result = llm.invoke(
                prompt.format(question=query, text=doc.page_content)
            )
            
            print(f"Raw LLM response: {result[:500]}...")
            
            # Parse the claims
            claim_lines = [line.strip() for line in result.split('\n') if line.strip().startswith('CLAIM:')]
            claims = [line.replace('CLAIM:', '').strip() for line in claim_lines]
            
            print(f"Document {i+1} generated {len(claims)} claims")
            
            # Create a new document for each claim with the original metadata
            for j, claim in enumerate(claims):
                if claim:  # Only add non-empty claims
                    print(f"  Claim {j+1}: {claim}")
                    claim_doc = Document(
                        page_content=claim,
                        metadata=doc.metadata
                    )
                    all_claims.append(claim_doc)
        except Exception as e:
            print(f"Error extracting claims from document {i+1}: {str(e)}")
            continue
    
    print(f"Total claims extracted: {len(all_claims)}")
    print("All extracted claims:")
    for i, claim in enumerate(all_claims):
        print(f"Claim {i+1}: {claim.page_content}")
    
    return all_claims

def rerank_claims(claims, query, top_k=5):
    """Rerank claims based on relevance to the query.
    
    Args:
        claims: List of Document objects containing claims
        query: The query to rerank claims for
        top_k: Number of top claims to return
        
    Returns:
        List of top_k reranked claims
    """
    if not claims:
        print("No claims to rerank.")
        return []
    
    print(f"Reranking {len(claims)} claims...")
    
    # Filter out claims containing "I do not know" or "I don't have enough information"
    filtered_claims = []
    for claim in claims:
        content = claim.page_content.lower()
        if "i do not know" not in content and "i don't have enough information" not in content:
            filtered_claims.append(claim)
        else:
            print(f"Filtered out claim: {claim.page_content[:100]}{'...' if len(claim.page_content) > 100 else ''}")
    
    print(f"After filtering, {len(filtered_claims)} claims remain.")
    
    # If no claims remain after filtering, return empty list
    if not filtered_claims:
        print("No claims remain after filtering.")
        return []
    
    # If we have fewer claims than top_k, return all filtered claims
    if len(filtered_claims) <= top_k:
        print(f"Fewer than {top_k} claims, returning all {len(filtered_claims)} claims.")
        return filtered_claims
    
    # Rerank based on embedding similarity to the query
    try:
        print("Reranking claims based on embedding similarity...")
        # Create embeddings for the query and all claims
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        
        query_embedding = embeddings.embed_query(query)
        
        # Calculate similarity scores
        claim_similarities = []
        for i, claim in enumerate(filtered_claims):
            claim_embedding = embeddings.embed_documents([claim.page_content])[0]
            similarity = np.dot(query_embedding, claim_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(claim_embedding)
            )
            claim_similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        claim_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top_k claims
        top_indices = [idx for idx, _ in claim_similarities[:top_k]]
        top_claims = [filtered_claims[idx] for idx in top_indices]
        
        print(f"Top {len(top_claims)} claims after reranking:")
        for i, claim in enumerate(top_claims):
            print(f"  Claim {i+1} (score: {claim_similarities[i][1]:.4f}): {claim.page_content[:100]}{'...' if len(claim.page_content) > 100 else ''}")
        
        return top_claims
    except Exception as e:
        print(f"Error reranking claims: {str(e)}")
        # Fall back to random selection if embedding-based reranking fails
        import random
        selected_claims = random.sample(filtered_claims, min(top_k, len(filtered_claims)))
        print(f"Falling back to random selection. Selected {len(selected_claims)} claims.")
        return selected_claims

class GraphRetriever:
    """A retriever that traverses a graph of documents."""
    
    def __init__(self, vectorstore, k=100, start_k=30, adjacent_k=20, max_depth=3):
        """
        Initialize the GraphRetriever.
        
        Args:
            vectorstore: The vectorstore to use for retrieval
            k: Total number of nodes to retrieve
            start_k: Number of initial nodes to retrieve
            adjacent_k: Number of adjacent nodes to retrieve per node
            max_depth: Maximum depth of graph traversal
        """
        self.vectorstore = vectorstore
        self.k = k
        self.start_k = start_k
        self.adjacent_k = adjacent_k
        self.max_depth = max_depth
    
    def get_relevant_documents(self, query, initial_docs=None):
        """
        Retrieve documents by traversing a graph.
        
        Args:
            query: The query to retrieve documents for
            initial_docs: Initial documents to start the traversal from
            
        Returns:
            List of documents
        """
        # If no initial docs provided, get them from the vectorstore
        if initial_docs is None:
            initial_docs = self.vectorstore.similarity_search(query, k=self.start_k)
        
        # Create a graph
        graph = nx.Graph()
        
        # Add initial nodes
        for i, doc in enumerate(initial_docs):
            graph.add_node(i, doc=doc)
        
        # Traverse the graph
        visited = set(range(len(initial_docs)))
        frontier = list(visited)
        all_docs = initial_docs.copy()
        
        # BFS traversal
        depth = 0
        while frontier and depth < self.max_depth and len(all_docs) < self.k:
            next_frontier = []
            
            for node_id in frontier:
                # Get the document
                doc = graph.nodes[node_id]['doc']
                
                # Find similar documents
                try:
                    similar_docs = self.vectorstore.similarity_search(
                        doc.page_content, k=self.adjacent_k
                    )
                    
                    # Add new documents to the graph
                    for similar_doc in similar_docs:
                        # Check if this is a new document
                        is_new = True
                        for existing_doc in all_docs:
                            if similar_doc.page_content == existing_doc.page_content:
                                is_new = False
                                break
                        
                        if is_new:
                            # Add to all docs
                            all_docs.append(similar_doc)
                            
                            # Add to graph
                            new_node_id = len(graph)
                            graph.add_node(new_node_id, doc=similar_doc)
                            graph.add_edge(node_id, new_node_id)
                            
                            # Add to next frontier
                            if new_node_id not in visited:
                                visited.add(new_node_id)
                                next_frontier.append(new_node_id)
                                
                                # Stop if we've reached the limit
                                if len(all_docs) >= self.k:
                                    break
                except Exception as e:
                    print(f"Error finding similar documents: {str(e)}")
                    continue
            
            # Update frontier
            frontier = next_frontier
            depth += 1
        
        return all_docs

class HybridGraphRetriever:
    """A hybrid retriever that combines vector and graph retrieval."""
    
    def __init__(
        self,
        vectorstore,
        embeddings,
        llm,
        k_docs: int = 5,
        graph_k: int = 100,
        graph_start_k: int = 30,
        graph_adjacent_k: int = 20,
        graph_max_depth: int = 3,
        rerank_top_k: int = 5
    ):
        """Initialize the hybrid retriever.
        
        Args:
            vectorstore: The vectorstore to use for retrieval
            embeddings: The embeddings model to use
            llm: The language model to use
            k_docs: Number of documents to retrieve from vector store
            graph_k: Total number of nodes to retrieve in graph traversal
            graph_start_k: Number of initial nodes to retrieve in graph traversal
            graph_adjacent_k: Number of adjacent nodes to retrieve per node
            graph_max_depth: Maximum depth of graph traversal
            rerank_top_k: Number of documents to keep after reranking
        """
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm
        self.k_docs = k_docs
        self.graph_k = graph_k
        self.graph_start_k = graph_start_k
        self.graph_adjacent_k = graph_adjacent_k
        self.graph_max_depth = graph_max_depth
        self.rerank_top_k = rerank_top_k
    
    def retrieve(self, query: str):
        """Retrieve documents using both vector and graph retrieval.
        
        Args:
            query: The query to retrieve documents for
            
        Returns:
            Dict with combined_docs, vector_docs, and graph_docs
        """
        # Vector retrieval
        print(f"Performing vector retrieval for query: {query}")
        vector_docs = self.vectorstore.similarity_search(query, k=self.k_docs)
        
        # Graph retrieval
        print(f"Performing graph retrieval for query: {query}")
        graph_docs = graph_retrieval(
            self.vectorstore,
            query,
            k=self.graph_k,
            start_k=self.graph_start_k,
            adjacent_k=self.graph_adjacent_k,
            max_depth=self.graph_max_depth
        )
        
        # Print content of graph_docs for analysis
        print(f"\nRetrieved {len(graph_docs)} documents from graph traversal:")
        for i, doc in enumerate(graph_docs[:3]):  # Show first 3 docs
            print(f"Document {i+1}/{len(graph_docs)}:")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"Content: {content_preview}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        if len(graph_docs) > 3:
            print(f"... and {len(graph_docs) - 3} more documents")
        
        # First rerank all documents to find the most relevant ones
        print("First combining and reranking all documents...")
        all_reranked_docs = self._combine_and_rerank(vector_docs, graph_docs, query)
        
        # Store original reranked documents for context preservation
        original_reranked_docs = all_reranked_docs.copy()
        
        # If LLM is provided, extract claims from the reranked documents
        if self.llm is not None:
            print("Extracting claims from reranked documents...")
            graph_claims = self._extract_claims_from_docs(all_reranked_docs, query, self.llm)
            
            # Create a combined result with both claims and original documents
            combined_result = {
                "combined_docs": graph_claims,  # Primary result is the claims
                "vector_docs": vector_docs,
                "graph_docs": graph_docs,
                "original_reranked_docs": original_reranked_docs  # Keep original docs for reference
            }
            return combined_result
        
        # If no LLM, just return the reranked documents
        return {
            "combined_docs": all_reranked_docs,
            "vector_docs": vector_docs,
            "graph_docs": graph_docs
        }
    
    def _extract_claims_from_docs(self, docs, query, llm):
        """Extract claims from documents using LLM."""
        # Group documents into communities using a simple approach
        communities = []
        doc_texts = [doc.page_content for doc in docs]
        
        print(f"Extracting claims from {len(docs)} documents...")
        
        # Simple community detection: group similar documents
        used_indices = set()
        
        for i, doc_text in enumerate(doc_texts):
            if i in used_indices:
                continue
                
            community = [docs[i]]
            used_indices.add(i)
            
            # Find similar documents to add to this community
            for j, other_text in enumerate(doc_texts):
                if j in used_indices:
                    continue
                    
                # Simple similarity check (can be improved)
                if len(set(doc_text.split()) & set(other_text.split())) > 10:
                    community.append(docs[j])
                    used_indices.add(j)
            
            communities.append(community)
        
        print(f"Grouped documents into {len(communities)} communities")
        
        # Extract claims from each community
        all_claims = []
        
        for i, community in enumerate(communities):
            print(f"Processing community {i+1}/{len(communities)} with {len(community)} documents...")
            # Join community documents
            community_text = "\n\n".join([doc.page_content for doc in community])
            
            # Use the LLM to extract claims
            claim_prompt = ChatPromptTemplate.from_template(
                """Extract the key claims or facts from the following text that are relevant to answering this question: {question}
                
                Text:
                {text}
                
                Return each claim on a new line, starting with '- '
                """
            )
            
            claims_text = llm.invoke(
                claim_prompt.format(question=query, text=community_text)
            )
            
            # Parse claims
            claims = []
            for line in claims_text.split("\n"):
                if line.strip().startswith("-"):
                    claim_text = line.strip()[2:].strip()
                    if claim_text:
                        # Create a document for each claim
                        claim_doc = Document(
                            page_content=claim_text,
                            metadata={"source": community[0].metadata.get("source", "Unknown")}
                        )
                        claims.append(claim_doc)
            
            print(f"Community {i+1} generated {len(claims)} claims")
            for j, claim in enumerate(claims):
                print(f"  Claim {j+1}: {claim.page_content[:100]}{'...' if len(claim.page_content) > 100 else ''}")
            
            all_claims.extend(claims)
        
        print(f"Total claims extracted: {len(all_claims)}")
        return all_claims
    
    def _combine_and_rerank(self, vector_docs, graph_docs, query):
        """Combine and rerank documents using embedding-based similarity."""
        # Combine documents
        combined_docs = vector_docs + graph_docs
        
        # Remove duplicates based on content
        unique_docs = []
        seen_contents = set()
        
        for doc in combined_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        print(f"Combined {len(vector_docs)} vector docs and {len(graph_docs)} graph docs into {len(unique_docs)} unique docs")
        
        # Use embedding-based reranking for better relevance assessment
        if not unique_docs:
            return []
        
        # Get embeddings for the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Get embeddings for all documents
        doc_contents = [doc.page_content for doc in unique_docs]
        
        # Embed all document contents at once for efficiency
        doc_embeddings = self.embeddings.embed_documents(doc_contents)
        
        # Calculate similarity scores
        similarities = []
        for doc_embedding in doc_embeddings:
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        # Create (index, similarity) pairs and sort by similarity in descending order
        doc_similarities = list(enumerate(similarities))
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top-k documents
        reranked_docs = []
        for idx, similarity in doc_similarities[:self.rerank_top_k]:
            doc = unique_docs[idx]
            # Add similarity score to metadata for debugging
            if hasattr(doc, "metadata"):
                doc.metadata["similarity_score"] = float(similarity)
            reranked_docs.append(doc)
        
        print(f"Reranked to top {len(reranked_docs)} documents")
        return reranked_docs

class RAGPipeline:
    def __init__(
        self,
        llm_model: str = "llama3.1:latest",
        embedding_model: str = "nomic-embed-text:latest",
        temperature: float = 0,
        k_docs: int = 5,
        use_reranking: bool = True,
        rerank_top_k: int = 5,
        use_sentence_window: bool = True,
        use_lazy_graph_rag: bool = False,
        use_hybrid_rag: bool = False,
        graph_k: int = 100,
        graph_start_k: int = 30,
        graph_adjacent_k: int = 20,
        graph_max_depth: int = 3,
        data_dir: str = "data",
        persist_dir: str = "chroma_db"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            llm_model (str): The LLM model to use
            embedding_model (str): The embedding model to use
            temperature (float): Temperature for generation
            k_docs (int): Number of documents to retrieve
            use_reranking (bool): Whether to use reranking
            rerank_top_k (int): Number of documents to keep after reranking
            use_sentence_window (bool): Whether to use sentence window chunking
            use_lazy_graph_rag (bool): Whether to use LazyGraphRAG
            use_hybrid_rag (bool): Whether to use Hybrid RAG
            graph_k (int): Number of total nodes to retrieve in graph traversal
            graph_start_k (int): Number of initial nodes to retrieve in graph traversal
            graph_adjacent_k (int): Number of adjacent nodes to retrieve per node
            graph_max_depth (int): Maximum depth of graph traversal
            data_dir (str): Directory containing documents
            persist_dir (str): Directory to persist the vector store
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.k_docs = k_docs
        self.use_reranking = use_reranking
        self.rerank_top_k = rerank_top_k
        self.use_sentence_window = use_sentence_window
        self.use_lazy_graph_rag = use_lazy_graph_rag
        self.use_hybrid_rag = use_hybrid_rag
        self.graph_k = graph_k
        self.graph_start_k = graph_start_k
        self.graph_adjacent_k = graph_adjacent_k
        self.graph_max_depth = graph_max_depth
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        
        # Initialize components to None
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.fallback_retriever = None
        self.qa_chain = None
        self.lazy_graph_prompt = None
        
        # Check if vectorstore exists
        self.vectorstore_ready = os.path.exists(self.persist_dir)
        print(f"Vectorstore ready: {self.vectorstore_ready}")

    def index_documents(self, force_reindex=False):
        """
        Index documents in the data directory.
        
        Args:
            force_reindex (bool): Whether to force reindexing
            
        Returns:
            str: Result message
        """
        print(f"Indexing documents from {self.data_dir}...")
        
        # Check if vectorstore already exists and we're not forcing reindex
        if os.path.exists(self.persist_dir) and not force_reindex:
            print("Index already exists. Use force_reindex=True to reindex.")
            self.vectorstore_ready = True
            return "Index already exists"
            
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Load documents
        try:
            # Set up loaders for different file types
            loaders = {
                ".txt": (TextLoader, {}),
                ".pdf": (PyPDFLoader, {})
            }
            
            # Create directory loader
            loader = DirectoryLoader(
                self.data_dir,
                glob="**/*.*",
                loader_cls=lambda file_path: loaders.get(
                    os.path.splitext(file_path)[1].lower(),
                    (TextLoader, {})  # Default to TextLoader
                )[0](file_path, **loaders.get(os.path.splitext(file_path)[1].lower(), (TextLoader, {}))[1])
            )
            
            # Load documents
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")
            
            # Split documents using a two-step approach for better semantic chunking
            # First, split into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            initial_splits = text_splitter.split_documents(documents)
            print(f"Initial split into {len(initial_splits)} chunks")
            
            # Then, apply sentence transformer token splitter for better semantic chunking
            if self.use_sentence_window:
                sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=20)
                splits = sentence_splitter.split_documents(initial_splits)
                print(f"Further split into {len(splits)} semantic chunks using sentence window")
            else:
                splits = initial_splits
                print(f"Using {len(splits)} chunks without sentence window")
            
            # Create vectorstore directly
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
                
            # Persist vectorstore
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
                
            self.vectorstore_ready = True
            return "Successfully indexed documents"
            
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return f"Error: {str(e)}"

    def initialize_components(self):
        """Initialize the RAG pipeline components.
        
        Returns:
            bool: Whether initialization was successful
        """
        try:
            print("Initializing embeddings with model:", self.embedding_model)
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            if self.vectorstore_ready:
                print("Loading vectorstore from:", self.persist_dir)
                self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            else:
                print("Vectorstore not ready. Please index documents first.")
                return False
            
            print("Initializing LLM with model:", self.llm_model)
            self.llm = OllamaLLM(model=self.llm_model, temperature=self.temperature)
            
            # Create fallback retriever for LazyGraphRAG
            self.fallback_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_docs})
            
            if self.use_lazy_graph_rag:
                print("Initializing LazyGraphRAG...")
                # Set up LazyGraphRAG prompt
                self.lazy_graph_prompt = ChatPromptTemplate.from_template("""
                You are a helpful AI assistant that answers questions based on the provided context.
                Use only the information from the context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """)
                
                # Initialize the LazyGraphRAG chain
                self._setup_lazy_graph_rag_chain()
                print("LazyGraphRAG initialized successfully")
                
            elif self.use_hybrid_rag:
                print("Initializing Hybrid RAG...")
                # Set up Hybrid RAG retriever
                self.retriever = HybridGraphRetriever(
                    vectorstore=self.vectorstore,
                    embeddings=self.embeddings,
                    llm=self.llm,
                    k_docs=self.k_docs,
                    graph_k=self.graph_k,
                    graph_start_k=self.graph_start_k,
                    graph_adjacent_k=self.graph_adjacent_k,
                    graph_max_depth=self.graph_max_depth,
                    rerank_top_k=self.rerank_top_k
                )
                
                # Set up Hybrid RAG chain
                self._setup_hybrid_rag_chain()
                print("Hybrid RAG initialized successfully")
                
            else:
                # Create standard retriever
                print("Initializing standard retriever...")
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_docs})
                
                # Set up standard QA chain
                print("Setting up standard QA chain...")
                self._setup_standard_qa_chain()
                
            print("Components initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            return False

    def _setup_standard_qa_chain(self):
        """
        Set up a standard QA chain.
        """
        # Create prompt template
        template = """
        You are a helpful AI assistant that answers questions based on the provided context.
        Use only the information from the context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the QA chain
        self.qa_chain = prompt | self.llm | StrOutputParser()

    def _setup_lazy_graph_rag_chain(self):
        """
        Set up the LazyGraphRAG chain.
        """
        # Create a function to process the query with LazyGraphRAG
        def _process_query(query):
            return process_lazy_graph_rag(
                query=query,
                vectorstore=self.vectorstore,
                llm=self.llm,
                prompt=self.lazy_graph_prompt,
                fallback_retriever=self.fallback_retriever,
                graph_k=self.graph_k,
                graph_start_k=self.graph_start_k,
                graph_adjacent_k=self.graph_adjacent_k,
                graph_max_depth=self.graph_max_depth,
                rerank_top_k=self.rerank_top_k
            )
        
        # Create prompt template
        template = """
        Answer the question based on the supporting claims.
        
        Only use information from the claims. Do not guess or make up any information.
        If the claims do not provide enough information to answer the question, say "I don't have enough information to answer this question."
        
        Supporting Claims:
        {claims}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the LazyGraphRAG chain
        self.lazy_graph_rag_chain = {
            "query": lambda x: x,
            "answer": lambda x: self.llm.invoke(
                prompt.format(claims=_process_query(x)["claims"], question=x)
            ),
            "sources": lambda x: _process_query(x)["sources"]
        }

    def _setup_hybrid_rag_chain(self):
        """
        Set up the Hybrid RAG chain.
        """
        # Create a function to process the query with Hybrid RAG
        def _process_query(query):
            result = self.retriever.retrieve(query)
            
            # Get claims from the combined_docs (which now contains claims)
            claims = [doc.page_content for doc in result["combined_docs"]]
            claims_text = "\n\n".join([f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)])
            
            # Get original documents if available
            original_docs = []
            if "original_reranked_docs" in result:
                original_docs = result["original_reranked_docs"]
                
            # Combine claims and original docs for context
            context_parts = []
            
            # Add claims section if we have claims
            if claims:
                context_parts.append(f"### Key Claims\n{claims_text}")
            
            # Add original documents section if we have them
            if original_docs:
                original_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(original_docs[:3])])
                context_parts.append(f"### Original Documents\n{original_text}")
            
            # Combine all context parts
            full_context = "\n\n" + "\n\n".join(context_parts)
            
            # Get sources
            sources = []
            for doc in result["combined_docs"]:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
            
            return {
                "context": full_context,
                "sources": sources
            }
        
        # Create prompt template
        template = """
        You are a helpful AI assistant that answers questions based on the provided context.
        Use the information from the context to answer the question. The context includes both key claims extracted from documents and some original document text.
        
        If you don't know the answer based on the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the Hybrid RAG chain
        self.hybrid_rag_chain = {
            "query": lambda x: x,
            "answer": lambda x: self.llm.invoke(
                prompt.format(context=_process_query(x)["context"], question=x)
            ),
            "sources": lambda x: _process_query(x)["sources"]
        }

    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            query: The question to answer
            
        Returns:
            Dict with answer and sources
        """
        try:
            print(f"Processing query: {query}")
            
            if not self.vectorstore_ready:
                return {"answer": "Error: Vectorstore not ready. Please index documents first.", "sources": []}
            
            if self.use_lazy_graph_rag:
                print("Using LazyGraphRAG to answer question")
                result = self.lazy_graph_rag_chain
                answer = result["answer"](query)
                sources = result["sources"](query)
                return {"answer": answer, "sources": sources}
                
            elif self.use_hybrid_rag:
                print("Using Hybrid RAG to answer question")
                result = self.hybrid_rag_chain
                answer = result["answer"](query)
                sources = result["sources"](query)
                return {"answer": answer, "sources": sources}
                
            else:
                print("Using standard QA chain to answer question")
                docs = self.retriever.get_relevant_documents(query)
                
                # Get sources
                sources = []
                for doc in docs:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.append(doc.metadata["source"])
                
                # Format documents for context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Generate answer
                answer = self.qa_chain.invoke({"context": context, "question": query})
                
                return {"answer": answer, "sources": sources}
                
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return {"answer": f"Error: {str(e)}", "sources": []}

    def _rerank_documents(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        """Custom reranking method to replace ContextualCompressionRetriever.
        
        Args:
            query: The query to rerank documents for
            docs: The list of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of top_k reranked documents
        """
        if not docs:
            return []
        
        print(f"Reranking {len(docs)} documents to top {top_k}...")
        
        # Get embeddings for the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Get embeddings for all documents
        doc_embeddings = []
        doc_contents = []
        
        for doc in docs:
            doc_contents.append(doc.page_content)
        
        # Embed all document contents at once for efficiency
        doc_embeddings = self.embeddings.embed_documents(doc_contents)
        
        # Calculate similarity scores
        similarities = []
        for doc_embedding in doc_embeddings:
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        # Create (index, similarity) pairs and sort by similarity in descending order
        doc_similarities = list(enumerate(similarities))
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top_k documents
        top_indices = [idx for idx, _ in doc_similarities[:top_k]]
        
        # Remove duplicate content (simple approach)
        seen_content = set()
        unique_docs = []
        
        for idx in top_indices:
            content = docs[idx].page_content
            # Only add if we haven't seen very similar content before
            if content not in seen_content:
                unique_docs.append(docs[idx])
                seen_content.add(content)
        
        print(f"Reranked to top {len(unique_docs)} documents")
        return unique_docs
