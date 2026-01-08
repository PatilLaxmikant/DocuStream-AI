# 📚 DocuStream AI

**DocuStream AI** is a production-ready Retrieval-Augmented Generation (RAG) application built with **Google Gemini**, **Qdrant Cloud**, and **Streamlit**. It allows users to upload PDF documents and chat with them using advanced AI, featuring real-time streaming responses, source citations, and enterprise-grade security.

## 🚀 Features

*   **⚡ Real-Time Streaming**: Chat responses stream character-by-character for a fluid experience.
*   **🔍 Smart Indexing & Duplicate Prevention**: Calculates file hashes to prevent re-indexing the same document twice, saving storage and cost.
*   **📖 Precision Citations**: Every answer provides expandable source citations, showing the exact page number and text snippet used.
*   **🔒 Secure Authentication**: Protected by a login system (default: `admin` / `password123`).
*   **☁️ Cloud Native**: Power by **Qdrant Cloud** for vector storage and **NeonDB (Postgres)** for data persistence.
*   **🛠️ Database Management**: Includes a "Power Wash" feature to clear vector data directly from the UI.

## 🛠️ Tech Stack

*   **LLM & Embeddings**: Google Gemini 2.5 Flash (`langchain-google-genai`)
*   **Vector Database**: Qdrant Cloud (`qdrant-client`)
*   **Frontend**: Streamlit
*   **Authentication**: Streamlit-Authenticator
*   **Database**: PostgreSQL (NeonDB)

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/PatilLaxmikant/DocuStream-AI.git
cd DocuStream-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.env` file in the root directory:
```ini
GEMINI_API_KEY="your_gemini_key"
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_key"
POSTGRES_URL="your_postgres_url"
```

### 4. Run the App
```bash
streamlit run streamlit_app.py
```
**Login credentials**:
*   **Username**: `admin`
*   **Password**: `password123`

## ☁️ Deployment (Streamlit Cloud)

1.  Push this repo to GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io).
3.  Deploy the app pointing to `streamlit_app.py`.
4.  Add your secrets (API Keys) in the Streamlit Cloud dashboard under **Advanced Settings**.

---
*Built with ❤️ by DocuStream AI Team*
