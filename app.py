import streamlit as st

# Page Config
st.set_page_config(
    page_title="AI-Powered Enterprise Chatbot", 
    page_icon="🤖", 
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
st.markdown('<h1 class="main-title">🤖 AI-Powered Enterprise Chatbot</h1>', unsafe_allow_html=True)
# st.markdown(
#     '''
#     <p class="subtitle">
#         Transform your workplace communication with our intelligent, multimodal chatbot platform. 
#         Featuring advanced HR & IT support, PDF querying capabilities, and RAG-based retrieval 
#         with comprehensive chat history management.
#     </p>
#     ''',
#     unsafe_allow_html=True
# )

# Navigation Cards
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

with col1:
    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    if st.button("🎯\n\nMultimodal HR & IT\nChatbot", key="page1"):
        st.switch_page("pages/pages1.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    if st.button("📊\n\nPDF Query\nSystem", key="page2"):
        st.switch_page("pages/pages2.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    if st.button("🚀\n\nRAG + Chat\nHistory", key="page3"):
        st.switch_page("pages/pages3.py")
    st.markdown('</div>', unsafe_allow_html=True)

# Feature Highlights
st.markdown(
    '''
    <div class="feature-highlight fade-in">
        <h3 style="color: #2c3e50; font-weight: 600; margin-bottom: 20px;">🌟 Platform Features</h3>
        <div class="feature-grid">
            <div class="feature-item">
                <h4>🔍 Intelligent Search</h4>
                <p>Advanced semantic search capabilities with context-aware responses</p>
            </div>
            <div class="feature-item">
                <h4>📁 Document Processing</h4>
                <p>Extract and query information from PDFs and various document formats</p>
            </div>
            <div class="feature-item">
                <h4>💬 Conversational AI</h4>
                <p>Natural language processing with memory and context retention</p>
            </div>
            <div class="feature-item">
                <h4>🔐 Enterprise Security</h4>
                <p>Secure, scalable architecture designed for enterprise environments</p>
            </div>
            <div class="feature-item">
                <h4>📈 Analytics Dashboard</h4>
                <p>Comprehensive insights and usage analytics for optimization</p>
            </div>
            <div class="feature-item">
                <h4>⚡ Real-time Processing</h4>
                <p>Lightning-fast responses with advanced caching mechanisms</p>
            </div>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
