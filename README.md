# 📄 Chat with Your PDF using LLMs

A powerful Streamlit app that allows you to **interact with your PDF files** using **LangChain**, **FAISS**, and **Hugging Face Transformers**. Upload a PDF and start asking questions — the app intelligently retrieves answers using embeddings and large language models.

---

## 🚀 Features

- 📂 Upload any PDF and start chatting instantly  
- 🤖 Uses Hugging Face Transformers for natural language understanding  
- 🔍 FAISS-powered vector search for fast information retrieval  
- 🧠 Retrieval-Augmented Generation (RAG) with LangChain  
- 💬 Clean and simple Streamlit interface

---

## 🧠 Tech Stack

- **Frontend**: Streamlit  
- **LLM**: Hugging Face Transformers  
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)  
- **Vector DB**: FAISS  
- **Framework**: LangChain  
- **Deployment**: Hugging Face Spaces

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf

# Install dependencies
pip install -r requirements.txt

# Set your Hugging Face token (required)
export HUGGINGFACEHUB_API_TOKEN=your_token_here

# Run the app
streamlit run app.py
