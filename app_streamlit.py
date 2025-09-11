# app_streamlit.py
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="EEG UI (embedded)", layout="wide")
st.title("Embedded EEG UI")

html = Path("eeg_ui.html").read_text(encoding="utf-8")
components.html(html, height=900, scrolling=True)
