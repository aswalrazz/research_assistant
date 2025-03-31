import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import streamlit as st
import base64


def encode_pdf(file_content):
    return base64.b64encode(file_content).decode("utf-8")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = None
    if "notes" not in st.session_state:
        st.session_state.notes = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-1.5-flash"
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "start_analysis" not in st.session_state:
        st.session_state.start_analysis = False
