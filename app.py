import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup the Page Configuration
st.set_page_config(page_title="Intelligent Lecture Notes Assistant", layout="centered")
st.title("📚 Intelligent Lecture Notes Assistant")

# Fetch the Groq API key from environment variables (used for Streamlit Secrets later)
groq_api_key = os.environ.get('GROQ_API_KEY')

# Fallback: if the key isn't found in secrets, show an input box in the sidebar
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if groq_api_key:
    # 2. Initialize Models
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # 3. File Upload Interface
    upload = st.file_uploader("Upload your lecture notes PDF file", type=["pdf"])

    if upload is not None:
        # Save the uploaded file locally so PyPDFLoader can process it
        with open("temp.pdf", "wb") as f:
            f.write(upload.getbuffer())
        
        with st.spinner("Processing PDF and building Vector Database... This might take a moment."):
            # 4. Processing logic: Load, Split, and Embed
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
        st.success(f"Vector Database created successfully with {len(docs)} text chunks! Ask away.")
        
        # 5. Set up the LangChain QA Prompt
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context. 
            If you do not know the answer based on the context, say so.
            
            Context: {context}
            
            Question: {input}
            """
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # 6. Chat Interface
        user_query = st.text_input("Ask a question about your notes:")
        
        if user_query:
            with st.spinner("Analyzing context and generating your answer..."):
                response = retrieval_chain.invoke({"input": user_query})
                st.markdown("### 🤖 Answer:")
                st.write(response["answer"])
else:
    st.sidebar.warning("⚠️ Please provide your Groq API Key to begin.")
    st.info("👋 To get started, please provide a Groq API Key in the sidebar or add it to your Streamlit Secrets.")