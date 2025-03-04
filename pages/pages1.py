
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

st.title("Enterprise Support AI Assistant")
st.write("Ask HR and IT-related queries.")
input_method = st.radio("Select input type:", ["Text", "Voice"])
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



api_key =os.getenv("GROQ_API_KEY")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Load and chunk documents
    def load_and_store_documents():
        document_paths = ["policy.pdf"]  # Predefined document list
        documents = []
        for doc_path in document_paths:
            loader = PyPDFLoader(doc_path)
            documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        return Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    
    vectorstore = load_and_store_documents()
    retriever = vectorstore.as_retriever()

    # Contextualizing system prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, reformulate a standalone question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # HR and IT Support-specific system prompt
    system_prompt = (
        "You are an AI assistant for HR and IT support. Use retrieved documents to provide responses. "
        "If frustration or hostility is detected in the user's query, suggest seeking human support. "
        "End each conversation by asking for feedback, which will be used to improve responses."
        "\n\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
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
    
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},  
        )
        st.write("Assistant:", response['answer'])
    
    # st.success(f"Audio recorded and saved as {audio_path}")
        # st.write("Chat History:", session_history.messages)
    
    feedback = st.text_area("Provide feedback to improve responses:")
    if st.button("Submit Feedback") and feedback:
        st.session_state.store[session_id].messages.append(("human", feedback))
        st.success("Feedback submitted successfully!")

else:
    st.warning("Please enter the Groq API Key")


if st.button("🏠 Go to Home Page"):
    st.switch_page("app.py")  # Ensure "Home.py" is the correct file name
