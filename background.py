import streamlit as st

def apply_background():
    st.markdown(
        """
        <style>

        /* =================================================
           MAIN APP BACKGROUND + PAGE TRANSITION
           ================================================= */
        .stApp {
            animation: fadeIn 0.8s ease-in-out;
            background: linear-gradient(
                135deg,
                #0a0a0a,
                #0b0000
            );
            color: #ffffff;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* =================================================
           STICKY GLASS HEADER
           ================================================= */
        header[data-testid="stHeader"] {
            position: sticky;
            top: 0;
            z-index: 999;
            background: rgba(15, 0, 0, 0.65);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255,0,0,0.1);
        }

        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg {
            color: #ff3333 !important;
            fill: #ff3333 !important;
        }

        /* =================================================
           SIDEBAR
           ================================================= */
        section[data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #1a0000,
                #330000
            );
            position: relative;
        }

        /* Sidebar animated divider */
        section[data-testid="stSidebar"]::after {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 3px;
            height: 100%;
            background: linear-gradient(
                180deg,
                transparent,
                #ff0000,
                transparent
            );
            animation: glowDivider 3s infinite;
        }

        @keyframes glowDivider {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }

        /* =================================================
           🔥 REMOVE BASEWEB FOCUS RECTANGLE
           ================================================= */
        *:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Sidebar dropdowns / select / multiselect */
        div[data-baseweb="select"] > div {
            background-color: #0a0a0a !important; /* pure black */
            border: 1px solid rgba(255,0,0,0.3) !important;
            color: #ffffff !important;
        }

        /* Highlighted/focused item in dropdown */
        div[data-baseweb="select"] li {
            background-color: #330000 !important; /* dark red for hover selection */
            color: #ff6666 !important;
        }

        /* Remove focus shadow for dropdown */
        div[data-baseweb="select"] div:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Multiselect selected items (tags/pills) */
        div[data-baseweb="tag"] {
            background-color: #ff1a1a !important;
            color: #fff !important;
            border: none !important;
        }

        div[data-baseweb="select"] span {
            outline: none !important;
        }

        /* =================================================
           PAGE SECTION CARDS
           ================================================= */
        .section-card {
            background: rgba(255,0,0,0.04);
            border-radius: 18px;
            padding: 25px;
            margin: 25px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            animation: cardFade 0.7s ease;
        }

        @keyframes cardFade {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* =================================================
           BUTTONS
           ================================================= */
        .stButton > button {
            background: linear-gradient(135deg, #ff0000, #660000);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.25s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.03);
            box-shadow: 0px 8px 20px rgba(255,0,0,0.35);
        }

        /* =================================================
           METRICS & DATAFRAME
           ================================================= */
        div[data-testid="metric-container"] {
            background: rgba(255,0,0,0.05);
            border-radius: 14px;
            padding: 15px;
        }

        .stDataFrame {
            background-color: rgba(255,0,0,0.03);
            border-radius: 12px;
            padding: 0.5rem;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


def section_card(title: str, icon: str = "📊"):
    st.markdown(
        f"""
        <div class="section-card">
            <h2>{icon} {title}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
