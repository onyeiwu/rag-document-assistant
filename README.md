# 📚 RAG Document Assistant

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Groq](https://img.shields.io/badge/Groq-LLM-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

## 🌟 What is This App?

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask natural language questions, with answers grounded in the document and supported by page-level references.

## Why this project matters
This project demonstrates how modern AI systems can combine retrieval and generation to reduce hallucinations and provide accurate, document-grounded answers.

## 🚀 Live Demo

👉 [Click here to use the app](https://your-app-link.streamlit.app)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 Upload Multiple PDFs | Upload one or more PDF files at once |
| 📋 Document Summary | Auto-generates a summary of your document |
| 💡 Question Suggestions | Suggests 5 relevant questions to ask |
| 💬 AI Chat | Ask any question about your document |
| 🎯 Confidence Score | Shows how reliable each answer is |
| ⬇️ Download Chat | Download your full conversation |
| 🧹 Clear Chat | Reset conversation anytime |
| 🗑️ Remove Documents | Clear all documents and start fresh |
| 🔄 Auto Model Fallback | Switches AI models if rate limit hit |

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Streamlit | Web application framework |
| PyMuPDF | PDF text extraction |
| Sentence Transformers | Text embeddings all-MiniLM-L6-v2 |
| FAISS | Vector similarity search |
| Groq | LLM API for generating answers |
| Python | Core programming language |

## 🔧 How It Works

- User uploads PDF
- Text is extracted using PyMuPDF
- Text is cleaned and filtered
- Text is split into chunks (300 words)
- Each chunk is converted into embeddings (384-dim vector)
- Stored in FAISS vector database
- User asks a question
- Question is embedded
- FAISS retrieves relevant chunks
- Context + question sent to Groq LLM
- AI generates answer with page references

## 📦 Installation — Run Locally

Step 1 — Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

Step 2 — Install dependencies:

pip install -r requirements.txt

Step 3 — Create a .env file and add this inside:

GROQ_API_KEY=your_groq_api_key_here

Step 4 — Run the app:

streamlit run app.py

## 🔑 Getting a Free Groq API Key

1. Go to https://console.groq.com
2. Create a free account
3. Click API Keys in the sidebar
4. Click Create API Key
5. Copy your key and add it to your .env file

## 🤖 AI Models Used

The app automatically switches between these models if one hits the rate limit:

| Model | Provider |
|-------|---------|
| openai/gpt-oss-20b | Groq |
| openai/gpt-oss-120b | Groq |
| groq/compound-mini | Groq |
| qwen/qwen3.6-27b | Groq |

## 📊 Limitations

- Doesn’t support scanned PDFs (no OCR yet)
- Performance depends on chunk quality
- Limited by API rate limits

## 📁 Project Structure

app.py — Main Streamlit application
requirements.txt — Python dependencies
.gitignore — Files excluded from GitHub
README.md — Project documentation
rag_lesson.ipynb — Development notebook

## ⚠️ Important Notes

- Never share your .env file or API keys
- The rag_index folder is excluded from GitHub
- Free Groq API has daily token limits
- For best results upload text-based PDFs not scanned images

## 👨‍💻 Author

Onyeiwu Gabriel Chibuzor
Personal Project
Built with Python and Streamlit

## 📜 License

This project is for educational purposes.
All rights reserved 2026 Onyeiwu Gabriel Chibuzor
