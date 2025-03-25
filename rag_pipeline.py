import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import networkx as nx
from collections import defaultdict

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
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Simple implementation of graph-based retrieval
def graph_retrieval(vectorstore, query, k=100, start_k=30, adjacent_k=20, max_depth=3):
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
def process_lazy_graph_rag(query, vectorstore, llm, prompt, fallback_retriever=None, graph_k=100, graph_start_k=30, graph_adjacent_k=20, graph_max_depth=3, rerank_top_k=5):
    """Process a query using LazyGraphRAG.
    
    Args:
        query: The query to answer
        vectorstore: The vectorstore to use for retrieval
        llm: The language model to use
        prompt: The prompt template to use
        fallback_retriever: A fallback retriever to use if graph retrieval fails
        graph_k: Total number of nodes to retrieve
        graph_start_k: Number of initial nodes to retrieve
        graph_adjacent_k: Number of adjacent nodes to retrieve per node
        graph_max_depth: Maximum depth of graph traversal
        rerank_top_k: Number of top claims to keep after reranking
        
    Returns:
        Dict with answer and claims
    """
    try:
        # Get documents using graph retrieval
        docs = graph_retrieval(
            vectorstore, 
            query, 
            k=graph_k, 
            start_k=graph_start_k, 
            adjacent_k=graph_adjacent_k, 
            max_depth=graph_max_depth
        )
        
        # Group documents into communities using a simple approach
        communities = []
        doc_texts = [doc.page_content for doc in docs]
        
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
        
        # Extract claims from each community
        all_claims = []
        
        for community in communities:
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
            
            all_claims.extend(claims)
        
        # Rank claims by relevance to the query
        ranked_claims = sorted(
            all_claims,
            key=lambda x: len(set(query.lower().split()) & set(x.page_content.lower().split())),
            reverse=True
        )
        
        # Apply reranking to limit to top-k claims
        if rerank_top_k > 0 and len(ranked_claims) > rerank_top_k:
            ranked_claims = ranked_claims[:rerank_top_k]
            print(f"Reranked to top {rerank_top_k} claims")
        
        # Format claims for the prompt
        claims_text = "\n\n".join([f"{i+1}. {claim.page_content}" for i, claim in enumerate(ranked_claims)])
        
        # Generate answer
        answer = llm.invoke(
            prompt.format(question=query, claims=claims_text)
        )
        
        return {"answer": answer, "claims": ranked_claims}
        
    except Exception as e:
        print(f"Error in LazyGraphRAG: {str(e)}")
        
        if fallback_retriever:
            # Use fallback retriever
            docs = fallback_retriever.get_relevant_documents(query)
            
            # Apply reranking to limit to top-k sources
            if rerank_top_k > 0 and len(docs) > rerank_top_k:
                docs = docs[:rerank_top_k]
                print(f"Reranked fallback to top {rerank_top_k} sources")
            
            # Format documents for the prompt
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer using standard approach
            fallback_prompt = ChatPromptTemplate.from_template(
                """Answer the question based on the following context:
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
            )
            
            answer = llm.invoke(
                fallback_prompt.format(question=query, context=context)
            )
            
            return {"answer": answer, "claims": docs}
        else:
            raise e

class RAGPipeline:
    def __init__(
        self,
        llm_model: str = "llama3.1:latest",
        embedding_model: str = "nomic-embed-text:latest",
        temperature: float = 0.7,
        k_docs: int = 5,
        use_reranking: bool = True,
        rerank_top_k: int = 5,
        use_sentence_window: bool = True,
        use_lazy_graph_rag: bool = False,
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
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            print(f"Split into {len(splits)} chunks")
            
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
        """
        Initialize the RAG pipeline components.
        
        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Initialize embeddings
            print(f"Initializing embeddings with model: {self.embedding_model}")
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            
            # Load vectorstore
            print(f"Loading vectorstore from: {self.persist_dir}")
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            
            # Get document count for debugging
            collection = self.vectorstore._collection
            doc_count = collection.count()
            print(f"Collection contains {doc_count} documents")
            
            # Create LLM with temperature setting
            print(f"Initializing LLM with model: {self.llm_model}")
            self.llm = OllamaLLM(model=self.llm_model, temperature=self.temperature)
            
            # Create the retriever based on configuration
            if self.use_lazy_graph_rag:
                print("Initializing LazyGraphRAG...")
                # Create base retriever for fallback
                self.fallback_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_docs})
                
                # Create prompt template for LazyGraphRAG
                self.lazy_graph_prompt = ChatPromptTemplate.from_template("""
                Answer the question based on the supporting claims.
                
                Only use information from the claims. Do not guess or make up any information.
                If the claims do not provide enough information to answer the question, say "I don't have enough information to answer this question."
                
                Supporting Claims:
                {claims}
                
                Question: {question}
                
                Answer:
                """)
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
        self.qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            query (str): The question to answer
            
        Returns:
            Dict[str, Any]: The answer and source documents
        """
        if not self.vectorstore_ready:
            return {
                "answer": "Error: Vectorstore not ready. Please index documents first.",
                "sources": []
            }
            
        if not self.vectorstore:
            return {
                "answer": "Error: Vectorstore not initialized. Please initialize components first.",
                "sources": []
            }
        
        try:
            if self.use_lazy_graph_rag:
                print(f"Answering question using LazyGraphRAG: {query}")
                try:
                    # Use LazyGraphRAG
                    result = process_lazy_graph_rag(
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
                    
                    # Extract answer and sources
                    answer = result["answer"]
                    claims = result["claims"]
                    
                    print(f"Answer: {answer}")
                    print(f"Used {len(claims)} claims")
                    
                    # Format sources for return
                    sources = []
                    for claim in claims:
                        sources.append({
                            "content": claim.page_content,
                            "metadata": claim.metadata
                        })
                    
                    return {"answer": answer, "sources": sources}
                    
                except Exception as e:
                    print(f"Error using LazyGraphRAG, falling back to standard retrieval: {str(e)}")
                    # Fall back to standard retrieval if LazyGraphRAG fails
                    if self.fallback_retriever:
                        # Use fallback retriever to get documents
                        docs = self.fallback_retriever.get_relevant_documents(query)
                        
                        # Create a simple prompt
                        prompt = ChatPromptTemplate.from_template("""
                        You are a helpful AI assistant that answers questions based on the provided context.
                        Use only the information from the context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer this question."
                        
                        Context:
                        {context}
                        
                        Question:
                        {question}
                        
                        Answer:
                        """)
                        
                        # Join document contents
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Generate answer
                        answer = self.llm.invoke(prompt.format(context=context, question=query))
                        
                        # Format sources for return
                        sources = []
                        for doc in docs:
                            sources.append({
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            })
                        
                        return {"answer": answer, "sources": sources}
                    else:
                        return {
                            "answer": "Error: LazyGraphRAG failed and no fallback retriever available.",
                            "sources": []
                        }
            else:
                print(f"Answering question using standard QA chain: {query}")
                # Use standard QA chain
                answer = self.qa_chain.invoke(query)
                
                # Get sources from retriever
                docs = self.retriever.get_relevant_documents(query)
                
                # Format sources for return
                sources = []
                for doc in docs:
                    sources.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                
                return {"answer": answer, "sources": sources}
        
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }
