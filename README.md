# 📚 Intelligent Lecture Notes Assistant

> An AI-powered study companion that allows students to upload lecture PDFs and interact with their notes using Retrieval-Augmented Generation (RAG).

---

## 🚀 Project Overview

The **Intelligent Lecture Notes Assistant** is a Streamlit-based web application that helps students study more efficiently by answering questions directly from uploaded lecture notes.

Instead of reading entire PDFs, users can ask questions in natural language and receive accurate, context-aware answers generated using a Large Language Model (LLM).

The application combines **RAG (Retrieval-Augmented Generation)**, **FAISS**, **LangChain**, **HuggingFace Embeddings**, and **Groq LLM** to provide intelligent responses.

---

## ✨ Features

- 📄 Upload Lecture PDFs
- 🤖 AI-powered Question Answering
- 📝 Generate Summaries
- ❓ Generate Quiz Questions
- ✅ Generate MCQs
- 🎤 Generate Viva Questions
- 🗂 Generate Flashcards
- 📚 Topic-wise Explanations
- ⚡ Fast semantic search using FAISS

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Interface |
| LangChain | RAG Pipeline |
| FAISS | Vector Database |
| HuggingFace Embeddings | Text Embeddings |
| Groq API | Large Language Model |
| PyPDFLoader | PDF Processing |

---

## 🏗️ Project Architecture

```
          User
            │
            ▼
      Upload PDF
            │
            ▼
     Extract PDF Text
            │
            ▼
      Split into Chunks
            │
            ▼
Generate Embeddings
            │
            ▼
 Store in FAISS Vector DB
            │
            ▼
 Retrieve Relevant Chunks
            │
            ▼
       Groq LLM
            │
            ▼
     Generated Response
```

---

## 📂 Project Structure

```
Intelligent-Lecture-Notes-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── lecture_notes/
├── vector_store/
└── assets/
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/intelligent-lecture-notes-assistant.git

cd intelligent-lecture-notes-assistant
```

---

### 2️⃣ Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create a **.env** file

```text
GROQ_API_KEY=your_api_key_here
```

---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 💻 How It Works

1. Upload one or more lecture PDFs.
2. The application extracts text from the PDFs.
3. Text is divided into smaller chunks.
4. HuggingFace generates embeddings.
5. Embeddings are stored inside FAISS.
6. Relevant chunks are retrieved based on the user's question.
7. Groq LLM generates an accurate answer using retrieved context.

---

## 📸 Screenshots

### Home Page

*(Add Screenshot Here)*

---

### Upload PDF

*(Add Screenshot Here)*

---

### Question Answering

*(Add Screenshot Here)*

---

### Summary Generation

*(Add Screenshot Here)*

---

### Quiz Generation

*(Add Screenshot Here)*

---

## 🎯 Applications

- Educational AI
- Smart Learning
- College Students
- Exam Preparation
- Digital Classrooms
- Self-paced Learning

---

## ✅ Advantages

- Saves study time
- Accurate context-based answers
- Easy-to-use interface
- Interactive learning experience
- AI-powered revision assistant

---

## 🔮 Future Enhancements

- Voice-based interaction
- Multiple PDF support
- OCR support for scanned PDFs
- Multi-language support
- Student progress tracking
- Mobile application

---

## 👨‍💻 Team

**Team 5**

**Project:** Intelligent Lecture Notes Assistant

---

## 📜 License

This project is developed for educational purposes.
