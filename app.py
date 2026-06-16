Here's a notebook named app.py
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.set_page_config(page_title="Intelligent Lecture Notes Assistant")
st.title("📚 Intelligent Lecture Notes Assistant")

# ── Session state init ──────────────────────────────────────────────
if "chains" not in st.session_state:
    st.session_state.chains = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar: setup ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")
    groq_api_key = st.text_input("Groq API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    chunk_size = st.selectbox("Chunk size", [500, 1000, 1500], index=1)
    k = st.selectbox("Chunks to retrieve (k)", [3, 5, 8], index=1)
    build_btn = st.button("Build Knowledge Base")

# ── Build chains on button click ────────────────────────────────────
if build_btn:
    if not groq_api_key or not uploaded_files:
        st.sidebar.error("Please provide API key and at least one PDF.")
    else:
        with st.spinner("Loading PDFs and building vector store..."):
            os.environ["GROQ_API_KEY"] = groq_api_key
            documents = []
            for uf in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})

            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            def make_chain(prompt_template):
                return (
                    {"context": retriever | format_docs, "question": lambda x: x}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )

            # Your existing prompts (unchanged)
            qa_prompt = ChatPromptTemplate.from_template("""
Hello! I'm your Lecture Notes Assistant. Answer using *only* the provided context.
If I can't find the answer, say: "Sorry, I couldn't find this in the lecture notes."
Context: {context}
Question: {question}
Answer:""")

            summary_prompt = ChatPromptTemplate.from_template("""
Give a clear and concise summary of these lecture notes.
Context: {context}
Summary:""")

            mcq_prompt = ChatPromptTemplate.from_template("""
Generate 5 multiple-choice questions (4 options each, mark correct answer).
Context: {context}
MCQs:""")

            quiz_prompt = ChatPromptTemplate.from_template("""
Generate 5 short-answer quiz questions.
Context: {context}
Quiz Questions:""")

            viva_prompt = ChatPromptTemplate.from_template("""
Generate 5 conceptual viva questions that go beyond simple recall.
Context: {context}
Viva Questions:""")

            flashcard_prompt = ChatPromptTemplate.from_template("""
Extract 10 key terms and definitions as 'Term: Definition' on separate lines.
Context: {context}
Flashcards:""")

            topic_prompt = ChatPromptTemplate.from_template("""
Provide a detailed explanation for the given topic using the lecture notes.
Context: {context}
Topic: {question}
Explanation:""")

            st.session_state.chains = {
                "qa": make_chain(qa_prompt),
                "summary": make_chain(summary_prompt),
                "mcq": make_chain(mcq_prompt),
                "quiz": make_chain(quiz_prompt),
                "viva": make_chain(viva_prompt),
                "flashcard": make_chain(flashcard_prompt),
                "topic": make_chain(topic_prompt),
            }
            st.session_state.chat_history = []
        st.sidebar.success(f"✅ Loaded {len(documents)} pages, {len(docs)} chunks.")

# ── Chat UI ──────────────────────────────────────────────────────────
if st.session_state.chains:
    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Mode selector + input
    mode = st.selectbox("Mode", [
        "Ask a Question", "Summary", "MCQs",
        "Quiz Questions", "Viva Questions", "Flashcards", "Topic Explanation"
    ])

    user_input = st.chat_input("Type your question or topic here...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        chains = st.session_state.chains
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                mode_map = {
                    "Ask a Question":    (chains["qa"],       user_input),
                    "Summary":           (chains["summary"],  "Generate a summary of the lecture notes."),
                    "MCQs":              (chains["mcq"],      "Generate MCQs from the key concepts."),
                    "Quiz Questions":    (chains["quiz"],     "Generate quiz questions from the key concepts."),
                    "Viva Questions":    (chains["viva"],     "Generate viva questions from the key concepts."),
                    "Flashcards":        (chains["flashcard"],"Extract key terms and definitions."),
                    "Topic Explanation": (chains["topic"],    user_input),
                }
                chain, query = mode_map[mode]
                result = chain.invoke(query)
                st.markdown(result)
        st.session_state.chat_history.append({"role": "assistant", "content": result})

else:
    st.info("👈 Upload PDFs and click **Build Knowledge Base** to get started.")
