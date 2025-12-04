import streamlit as st
import pandas as pd

st.set_page_config(page_title="Recruitment Dashboard", layout="wide")

# ======================
#       CUSTOM CSS
# ======================
st.markdown("""
<style>
    .stTabs [role="tablist"] {
        display: flex;
        gap: 3rem;
        justify-content: center;
        border-bottom: 4px solid #2F4AFB;
        padding-bottom: 14px;
        margin-top: 40px;
    }
    .stTabs [role="tab"] {
        padding: 26px 48px !important;
        border-radius: 18px 18px 0 0;
        background: #f5f7ff00;
    }

    .stTabs [role="tab"] p {
        font-size: 30px !important; 
        font-weight: 900 !important;
        margin: 0px !important;
        letter-spacing: 1px;
        color: #444 !important;
    }

    /* Hover */
    .stTabs [role="tab"]:hover p {
        color: #2F4AFB !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2F4AFB !important;
        box-shadow: 0px 8px 18px rgba(0,0,0,0.25);
        border-bottom: 5px solid #0E1E99;
    }
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }

    .title-clean {
        font-size: 80px !important;   /* <<< dibesarkan kembali */
        font-weight: 900 !important;
        text-align: center;
        color: #3A6DFA;
        margin-top: 25px;
        margin-bottom: 35px;
    }

</style>
""", unsafe_allow_html=True)


st.markdown("<div class='title-clean'>DATA ALCHEMIST</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📊 Predict Optimal Score", "📊 Predict New Dataset"])

with tab1:
    from views import dashboard
    dashboard.run()

with tab2:
    from views import predictpage
    predictpage.run()

with tab3:
    from views import predictup
    predictup.run()