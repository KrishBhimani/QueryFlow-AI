import streamlit as st

# Page Config
st.set_page_config(page_title="AI-Powered Enterprise Chatbot", page_icon="🏠", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        height: 120px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        background-color: white;
        color: #2C3E50;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: white;
        transform: scale(1.05);
        color: black;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Description
st.markdown('<h1 class="title">AI-Powered Enterprise Chatbot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="description">An intelligent, multimodal chatbot for HR & IT support with PDF querying, RAG-based retrieval, and chat history management.</p>',
    unsafe_allow_html=True
)


# Create four larger cards using buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Multimodal HR & IT Chatbot", key="page1"):
        st.switch_page("pages/pages1.py")

with col2:
    if st.button("📊 PDF Query System", key="page2"):
        st.switch_page("pages/pages2.py")

with col3:
    if st.button("🤖 RAG + Chat History", key="page3"):
        st.switch_page("pages/pages3.py")

# with col4:
#     if st.button("⚙️ Advanced Features", key="page4"):
#         st.switch_page("pages/pages4.py")

