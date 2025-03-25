graph LR
    subgraph User
        A[User Asks Question] --> B(Streamlit UI)
    end

    subgraph RAG Process
        B --> C{Question}
        C --> D[ChromaDB Vectorstore]
        D --> E[Retrieve Relevant Documents]
        E --> F[Rerank Documents 
        - CrossEncoder]
        F --> G[Ollama LLM]
        G --> H(Generate Answer)
    end

     H --> I[Display Answer in Streamlit]

    subgraph Document Indexing
      J[PDF Documents] --> K[LangChain Document Loaders -
      UnstructuredFileLoader]
      K --> L[LangChain Text Splitters- 
      RecursiveCharacterTextSplitter -
       SentenceTransformersTokenTextSplitter ]
      L --> M[Ollama Embeddings 
      - nomic-embed-text]
      M --> D
    end

Proposed Structure:
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
