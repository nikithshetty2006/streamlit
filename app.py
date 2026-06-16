import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Intelligent Lecture Notes Assistant", layout="centered")
st.title("📚 Intelligent Lecture Notes Assistant")

# Fetch API key
groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # LOAD THE PRE-BUILT DATABASE (This replaces the PDF uploader)
    try:
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True # Required by LangChain to load .pkl files
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        st.success("Pre-built Vector Database loaded successfully!")
    except Exception as e:
        st.error("Could not find the faiss_index folder. Make sure the .faiss and .pkl files are uploaded to GitHub!")
        st.stop()
        
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
    
    user_query = st.text_input("Ask a question about the lecture notes:")
    
    if user_query:
        with st.spinner("Analyzing context and generating your answer..."):
            response = retrieval_chain.invoke({"input": user_query})
            st.markdown("### 🤖 Answer:")
            st.write(response["answer"])
else:
    st.sidebar.warning("⚠️ Please provide your Groq API Key to begin.")
