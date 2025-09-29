import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

# Page Config
st.set_page_config(
    page_title="RAG + Chat History", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px;
        margin: 20px auto;
        max-width: 1200px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        line-height: 1.2;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #495057;
        font-weight: 400;
        margin: 0 auto 50px auto;
        line-height: 1.6;
        max-width: 800px;
        padding: 0 40px;
        display: block;
        width: 100%;
    }
    
    /* Card container */
    .card-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        margin: 40px auto;
        flex-wrap: wrap;
        width: 100%;
        max-width: 1000px;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 280px !important;
        height: 140px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 20px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        color: #2c3e50 !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15) !important;
        position: relative !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        font-family: 'Inter', sans-serif !important;
        text-align: center !important;
        line-height: 1.3 !important;
        margin: 0 auto !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.01) !important;
    }
    
    /* Streamlit button container centering */
    .stButton {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    
    /* Feature highlights */
    .feature-highlight {
        text-align: center;
        margin-top: 60px;
        padding: 30px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 30px;
    }
    
    .feature-item {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .feature-item h4 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .feature-item p {
        color: #6c757d;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 36px;
        }
        
        .subtitle {
            font-size: 18px;
            padding: 0 20px;
        }
        
        .stButton > button {
            width: 100% !important;
            height: 140px !important;
        }
        
        .card-container {
            gap: 20px;
        }
    }
    
    /* Loading animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container
st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)

# Title and Description
st.markdown('<h1 class="main-title">🚀 RAG + Chat History</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload PDFs and chat with their content using persistent chat history.</p>', unsafe_allow_html=True)

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# api_key = st.text_input("Enter your Groq API key:", type="password")
api_key=os.getenv("GROQ_API_KEY")
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
                           "formulate a standalone question which can be understood without the chat history. "
                           "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
                           "to answer the question. If you don't know the answer, say that you don't know. "
                           "Use seven to eight sentences maximum and keep the answer concise. Also analyze the query "
                           "that the user adds and identify the sentiment. If frustration is detected, suggest the user seek human support."
                           "\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the Groq API Key")


# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

if st.button("🏠 Go to Home Page"):
    st.switch_page("app.py")
