# QueryFlow.AI - AI-Powered Enterprise Support Agent

An intelligent **AI-powered enterprise support assistant** designed to streamline employee queries across **HR, IT, and document-related tasks**. QueryFlow.AI integrates **multi-modal interactions**, **retrieval-augmented generation (RAG)**, and **conversational memory**, making it an indispensable tool for enterprise knowledge management.

## 🚀 Features

- **Multimodal HR & IT Chatbot**: Provides instant responses to HR and IT-related queries.
- **PDF Query System**: Upload PDFs and extract relevant insights using RAG-based retrieval.
- **RAG with Chat History**: Ensures context-aware responses by maintaining conversational memory.
- **Voice & Text Input**: Users can interact via both text and voice commands.
- **Secure & Scalable**: Built with API security and modular architecture for enterprise integration.
- **Modern UI**: Developed with **Streamlit** for an intuitive user experience.
- **Fast Information Retrieval**: Optimized with **vector stores** for efficient document search.

## 🛠️ Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/KrishBhimani/QueryFlow.AI.git
```

### 2️⃣ Create a Virtual Environment

#### For Windows:
```sh
python -m venv venv
venv\Scripts\activate
```

#### For macOS/Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```sh
streamlit run app.py
```

## 📌 Usage

1. Open the **Streamlit UI** in your browser.
2. Select a feature (Multimodal Chatbot, PDF Query, etc.).
3. Enter your query via text or voice.
4. Receive **context-aware AI-generated responses**.

## 🔧 Technologies Used

**Python, Flask, React, Streamlit, RAG (Retrieval-Augmented Generation), OpenAI/Groq Models, Whisper AI, Vector Stores, Hugging Face APIs, Torch, Vite, Tailwind CSS**

## 🚀 Challenges & Solutions

- **Handling Large-Scale Documents** → Optimized **vector stores** for faster retrieval.
- **Ensuring Real-Time Responses** → Implemented **efficient query processing & caching mechanisms**.
- **Seamless Frontend-Backend Sync** → Used **Flask + React** for smooth user experience.
- **Speech-to-Text Processing** → Integrated **Whisper AI** for accurate transcription.

## 🤝 Contributing

Contributions are welcome! Feel free to submit **issues** or **pull requests** to improve QueryFlow.AI.

---

