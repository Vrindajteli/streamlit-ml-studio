import streamlit as st
# (Keep your other imports)
import eda
import visualization
import training
import testing
from background import apply_background

# 1. Page Configuration
st.set_page_config(page_title="ML Studio", layout="wide")
apply_background()

# 2. Custom CSS (Corrected Sidebar and Red Theme)
st.markdown("""
    <style>
    /* SIDEBAR: Only target the FIRST widget label (Navigation) */
    [data-testid="stSidebar"] .stRadio > label:first-child p {
        font-size: 35px !important; 
        font-weight: 800 !important;
        color: #FFFFF !important; /* Red for emphasis */
        letter-spacing: 1px;
        line-height: 1.2;
    }

    /* Keep the actual radio options (Home, EDA, etc) at normal size */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 16px !important;
        font-weight: 400 !important;
    }

    /* Centered Container */
    .centered-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-top: 100px;
    }

    /* Red Pulsing Title */
    @keyframes red-pulse {
        from { opacity: 0.8; text-shadow: 0 0 2px #FF0000; }
        to { opacity: 1; text-shadow: 0 0 15px #FF0000, 0 0 25px #8B0000; }
    }

    .welcome-title {
        font-size: 4rem;
        font-weight: 800;
        color: #FF0000; /* Red Text */
        text-align: center;
        animation: red-pulse 2.5s infinite alternate;
    }

    /* Middle-to-Sides Flowing Red Line */
    @keyframes line-flow {
        0% { width: 0%; opacity: 0; }
        50% { width: 50%; opacity: 0.8; }
        100% { width: 0%; opacity: 0; }
    }

    .flowing-line {
        height: 2px;
        margin: 15px auto;
        background: linear-gradient(90deg, transparent, #FF0000, transparent);
        animation: line-flow 4s infinite ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "EDA", "Visualization", "Training", "Testing"]
)

# 4. Page Logic
if page == "Home":
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="welcome-title">Welcome to ML Studio</h1>', unsafe_allow_html=True)
    st.markdown('<div class="flowing-line"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:white; text-align:center;">Select a module from the sidebar to begin.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "EDA":
    eda.show()

elif page == "Visualization":
    visualization.show()

elif page == "Training":
    training.show()

elif page == "Testing":
    testing.show()
