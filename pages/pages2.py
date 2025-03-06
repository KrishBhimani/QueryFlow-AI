import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()
# os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')
# os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)
st.title("PDF Querying")
uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
def create_vector_embedding():
    st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.loader=PyPDFLoader(temppdf)
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.text=st.session_state.splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.text,st.session_state.embeddings)
# st.title("RAG Document Q&A With Groq And Llama3")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
user_prompt=st.text_input("Enter your query from the research paper")


import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

if st.button("🏠 Go to Home Page"):
    st.switch_page("app.py")  # Ensure "Home.py" is the correct file name