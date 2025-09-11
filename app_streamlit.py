# app_streamlit_embed.py
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="EEG Embedded", layout="wide")
st.title("EEG-UI (embedded)")
html = Path("eeg_ui.html").read_text(encoding="utf-8")
components.html(html, height=1000, scrolling=True)
