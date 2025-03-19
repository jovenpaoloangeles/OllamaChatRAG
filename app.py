import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from create_vectorstore import create_vectorstore, index_exists

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# App title and description
st.title("RAG Chatbot")

# Debug section
with st.expander("Debug Information", expanded=False):
    st.write("This section shows debugging information about the application.")
    if os.path.exists("chroma_db"):
        st.write(f"Chroma DB exists with {len(os.listdir('chroma_db'))} items")
    else:
        st.write("Chroma DB does not exist")
    
    if os.path.exists("data"):
        data_dirs = os.listdir("data")
        st.write(f"Data directory exists with {len(data_dirs)} items: {', '.join(data_dirs)}")
    else:
        st.write("Data directory does not exist")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    available_models = ["llama3.1", "gemma3:1b"]
    model_name = st.selectbox(
        "Select Ollama Model",
        available_models,
        index=0
    )
    
    # Temperature setting
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, 
                          help="Higher values make the output more random, lower values make it more deterministic")
    
    # Check if index exists
    index_status = index_exists("chroma_db")
    if index_status:
        st.success("Document index already exists")
        st.session_state["vectorstore_ready"] = True
        force_reindex = st.checkbox("Force reindex", value=False, help="Check this to reindex all documents even if an index already exists")
    else:
        st.warning("No document index found. Please index your documents.")
        force_reindex = False
    
    # Index documents button
    if st.button("Index Documents"):
        with st.spinner("Indexing documents..."):
            # Create vector store
            result = create_vectorstore(data_dir="data", persist_dir="chroma_db", force_reindex=force_reindex)
            
            if result.startswith("Successfully"):
                st.session_state["vectorstore_ready"] = True
                st.success(result)
            elif result.startswith("Index already exists"):
                st.info(result)
                st.session_state["vectorstore_ready"] = True
            else:
                st.error(result)

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if vectorstore is ready
    if "vectorstore_ready" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("Please index documents first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please index documents first."})
    else:
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Load embeddings and vectorstore
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
                
                # Get document count for debugging
                collection = vectorstore._collection
                doc_count = collection.count()
                with st.expander("Retrieval Debug", expanded=False):
                    st.write(f"Collection contains {doc_count} documents")
                    st.write(f"Query: {prompt}")
                
                # Create retriever with more results
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                
                # Retrieve documents for debugging
                retrieved_docs = retriever.get_relevant_documents(prompt)
                with st.expander("Retrieved Documents", expanded=False):
                    if not retrieved_docs:
                        st.warning("No documents were retrieved for this query.")
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"Document {i+1}:")
                        st.write(f"Content: {doc.page_content[:500]}...")
                        st.write(f"Metadata: {doc.metadata}")
                        st.write("---")
                
                # Create LLM with temperature setting
                llm = OllamaLLM(model=model_name, temperature=temperature)
                
                # Create prompt template with more detailed instructions
                template = """
                You are a helpful assistant that answers questions based on the provided context.
                
                Context:
                {context}
                
                Question: {question}
                
                Instructions:
                1. Answer the question based only on the provided context.
                2. If the answer cannot be found in the context, say "I don't have enough information to answer this question."
                3. Be concise and to the point.
                4. If the context contains relevant information but not a complete answer, use what's available to provide a partial answer.
                5. Format your answer in a readable way.
                
                Answer:
                """
                
                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template=template
                )
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                # Get response
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                
                # Display response
                message_placeholder.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
