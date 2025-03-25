import os
import argparse
from rag_pipeline import RAGPipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run RAG pipeline without UI")
    parser.add_argument("--llm", default="llama3.1:latest", help="LLM model to use")
    parser.add_argument("--embeddings", default="nomic-embed-text:latest", help="Embedding model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--k-docs", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--no-reranking", action="store_true", help="Disable reranking")
    parser.add_argument("--rerank-top-k", type=int, default=5, help="Number of documents to keep after reranking")
    parser.add_argument("--no-sentence-window", action="store_true", help="Disable sentence window chunking")
    parser.add_argument("--use-lazy-graph-rag", action="store_true", help="Enable LazyGraphRAG for retrieval")
    parser.add_argument("--graph-k", type=int, default=100, help="Number of total nodes to retrieve in graph traversal")
    parser.add_argument("--graph-start-k", type=int, default=30, help="Number of initial nodes to retrieve in graph traversal")
    parser.add_argument("--graph-adjacent-k", type=int, default=20, help="Number of adjacent nodes to retrieve per node")
    parser.add_argument("--graph-max-depth", type=int, default=3, help="Maximum depth of graph traversal")
    parser.add_argument("--data-dir", default="data", help="Directory containing documents")
    parser.add_argument("--persist-dir", default="chroma_db", help="Directory to persist the vector store")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing of documents")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("query", nargs="?", default=None, help="Query to answer (if not in interactive mode)")
    
    args = parser.parse_args()
    
    # Create the RAG pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        llm_model=args.llm,
        embedding_model=args.embeddings,
        temperature=args.temperature,
        k_docs=args.k_docs,
        use_reranking=not args.no_reranking,
        rerank_top_k=args.rerank_top_k,
        use_sentence_window=not args.no_sentence_window,
        use_lazy_graph_rag=args.use_lazy_graph_rag,
        graph_k=args.graph_k,
        graph_start_k=args.graph_start_k,
        graph_adjacent_k=args.graph_adjacent_k,
        graph_max_depth=args.graph_max_depth,
        data_dir=args.data_dir,
        persist_dir=args.persist_dir
    )
    
    # Check if index exists, if not create it
    if not pipeline.vectorstore_ready or args.force_reindex:
        print("Indexing documents...")
        pipeline.index_documents(force_reindex=args.force_reindex)
    
    # Initialize components
    print("Initializing components...")
    if not pipeline.initialize_components():
        print("Failed to initialize components. Exiting.")
        return
    
    print("RAG pipeline ready!")
    
    # Interactive mode
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter your question: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            # Get answer
            result = pipeline.answer_question(query)
            
            # Print answer
            print("\nAnswer:")
            print(result["answer"])
            
            # Print sources
            print("\nSources:")
            for i, source in enumerate(result["sources"]):
                source_name = source["metadata"].get("source", "Unknown")
                if isinstance(source_name, str) and len(source_name) > 50:
                    source_name = f"...{source_name[-50:]}"
                print(f"Source {i+1}: {source_name}")
    
    # Single query mode
    elif args.query:
        print(f"\nAnswering query: {args.query}")
        result = pipeline.answer_question(args.query)
        
        # Print answer
        print("\nAnswer:")
        print(result["answer"])
        
        # Print sources
        print("\nSources:")
        for i, source in enumerate(result["sources"]):
            source_name = source["metadata"].get("source", "Unknown")
            if isinstance(source_name, str) and len(source_name) > 50:
                source_name = f"...{source_name[-50:]}"
            print(f"Source {i+1}: {source_name}")
    
    else:
        print("No query provided. Use --interactive or provide a query.")

if __name__ == "__main__":
    main()
