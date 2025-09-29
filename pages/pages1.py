
import requests
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
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
    page_title="Multimodal HR & IT Chatbot", 
    page_icon="🎯", 
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
st.markdown('<h1 class="main-title">🎯 Enterprise Support AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask HR and IT-related queries.</p>', unsafe_allow_html=True)
input_method = st.radio("Select input type:", ["Text", "Voice"])
print("select input type")
if input_method == "Voice":
    recognizer = sr.Recognizer()

    if st.button("Record Voice"):
        with sr.Microphone() as source:
            st.write("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        # Save the audio file
        audio_path = "recorded_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio.get_wav_data())

        # Display the saved audio file in Streamlit
        st.audio(audio_path, format="audio/wav")
        # st.success(f"Audio recorded and saved as {audio_path}")


    def query(audio_file):
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
        headers = {"Authorization": "Bearer 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"}
        
        with open(audio_file, "rb") as f:
            data = f.read()
        
        response = requests.post(API_URL, headers=headers, data=data)
        
        if response.status_code == 200:
            return response.json().get("text", "No transcription available")
        else:
            return f"Error: {response.status_code}, {response.text}"

    # Get user input

    user_input = query("recorded_audio.wav")
    # st.write("Transcribed Text:", output_text)
if input_method == "Text":
    user_input = st.text_input("Your question:")



# Load HuggingFace Embeddings
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



os.environ['GROQ_API_KEY'] =os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

# Load and chunk documents
def load_and_store_documents():
    # First try to load the existing vectorstore
    try:
        print("Attempting to load existing vectorstore...")
        # Try with embedding_function parameter instead of embedding
        return Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    except Exception as e:
        print(f"Error loading existing vectorstore: {e}")
        
        # Try to repair the existing database by recreating it in place
        try:
            print("Attempting to repair existing vectorstore...")
            
            # Load documents
            document_paths = ["policy.pdf"]
            documents = []
            for doc_path in document_paths:
                loader = PyPDFLoader(doc_path)
                documents.extend(loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            
            # Try to create a temporary in-memory vectorstore first
            temp_vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            print("Successfully created temporary vectorstore")
            
            # Now try to create a persistent vectorstore
            return Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
        except Exception as e2:
            print(f"Error repairing vectorstore: {e2}")
            
            # If repair fails, try in-memory as last resort
            print("Falling back to in-memory vectorstore...")
            try:
                document_paths = ["policy.pdf"]
                documents = []
                for doc_path in document_paths:
                    loader = PyPDFLoader(doc_path)
                    documents.extend(loader.load())
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
                splits = text_splitter.split_documents(documents)
                # Create in-memory vectorstore without persistence
                return Chroma.from_documents(documents=splits, embedding=embeddings)
            except Exception as e3:
                print(f"Error creating in-memory vectorstore: {e3}")
                raise e3

# Add a flag to track if we're in a fallback mode
fallback_mode = False

try:
    vectorstore = load_and_store_documents()
    print("Vectorstore loaded successfully")
except Exception as e:
    st.error(f"Error loading vectorstore: {e}")
    st.warning("Falling back to simple mode without document retrieval.")
    vectorstore = None
    fallback_mode = True
# Only create retriever if vectorstore was loaded successfully
if vectorstore:
    try:
        retriever = vectorstore.as_retriever()
        print("Retriever created successfully")
    except Exception as e:
        st.error(f"Error creating retriever: {e}")
        retriever = None
else:
    retriever = None
# Contextualizing system prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, reformulate a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Only create history_aware_retriever if retriever was created successfully
if retriever:
    try:
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        print("History aware retriever created successfully")
    except Exception as e:
        st.error(f"Error creating history aware retriever: {e}")
        history_aware_retriever = None
else:
    history_aware_retriever = None

# HR and IT Support-specific system prompt
system_prompt = (
    "You are an AI assistant for HR and IT support. Use retrieved documents to provide responses. "
    "If frustration or hostility is detected in the user's query, suggest seeking human support. "
    "if any signs of self harm or harm to others is detected, immediately stop conversing with the user and output the message as it is: Sorry i cant help you eith this, seek human support" 
    "End each conversation by asking for feedback, which will be used to improve responses."
    "\n\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create question_answer_chain
try:
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    print("Question answer chain created successfully")
except Exception as e:
    st.error(f"Error creating question answer chain: {e}")
    question_answer_chain = None

# Only create rag_chain if history_aware_retriever was created successfully
if history_aware_retriever and question_answer_chain:
    try:
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        print("RAG chain created successfully")
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        rag_chain = None
else:
    rag_chain = None

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Only create conversational_rag_chain if rag_chain was created successfully
if rag_chain:
    try:
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        print("Conversational RAG chain created successfully")
    except Exception as e:
        st.error(f"Error creating conversational RAG chain: {e}")
        conversational_rag_chain = None
else:
    conversational_rag_chain = None

if user_input:
    if conversational_rag_chain:
        try:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},  
            )
            st.write("Assistant:", response['answer'])
        except Exception as e:
            st.error(f"Error processing your request: {e}")
            st.write("Assistant: I'm sorry, I encountered an error while processing your request. Please try again later.")
    elif fallback_mode:
        # In fallback mode, use the LLM directly without document retrieval
        try:
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant for HR and IT support. You can provide general information about HR policies and IT support, but you don't have access to specific company documents."),
                ("human", "{input}")
            ])
            
            chain = fallback_prompt | llm
            response = chain.invoke({"input": user_input})
            st.write("Assistant (Fallback Mode):", response.content)
        except Exception as e:
            st.error(f"Error in fallback mode: {e}")
            st.write("Assistant: I'm sorry, I'm experiencing technical difficulties. Please try again later.")
    else:
        st.error("The assistant is not fully initialized. Please check the error messages above.")
        st.write("Assistant: I'm sorry, I'm having trouble initializing my knowledge base. Please try again later.")

# st.success(f"Audio recorded and saved as {audio_path}")
    # st.write("Chat History:", session_history.messages)

feedback = st.text_area("Provide feedback to improve responses:")
if st.button("Submit Feedback") and feedback:
    st.session_state.store[session_id].messages.append(("human", feedback))
    st.success("Feedback submitted successfully!")




# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

if st.button("🏠 Go to Home Page"):
    st.switch_page("app.py")
