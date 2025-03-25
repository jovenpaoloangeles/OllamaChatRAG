graph LR
    subgraph User
        A[User Asks Question] --> B(Streamlit UI)
    end

    subgraph GraphRAG Process
        B --> C{Question}
        C --> D[Retrieve Context with GraphRAG RetrievalGraph]
        D --> E[Neo4j Knowledge Graph]
        D --> F[Query Graph with Cypher]
        D --> G[Rerank Nodes/Documents 
        - CrossEncoder]
        G --> H[Ollama LLM]
        H --> I(Generate Answer)
    end

    I --> J[Display Answer in Streamlit]

    subgraph Document Indexing
        K[PDF Documents] --> L[LangChain Document Loaders - 
        UnstructuredFileLoader]
        L --> M[LangChain Text Splitters - 
        RecursiveCharacterTextSplitter]
        M --> N[Ollama Embeddings - 
        nomic-embed-text]
        N --> O[GraphRAG Indexing - 
        Node Creation + Relations]
        O --> E
    end


Rewrite @app2.py to use GraphRAG (https://github.com/neo4j/neo4j-graphrag-python?tab=readme-ov-file) instead. Check @docs:langchain-python-docs . As it is another implementation, make sure to put all the files in @GraphRAGimplementation folder