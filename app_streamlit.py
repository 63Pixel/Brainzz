import os, io, zipfile, glob, re, tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from ftplib import FTP
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EEG-Auswertung", layout="wide")

st.title("EEG-Auswertung im Browser")
st.caption("Rohdaten → Bandanteile → Stress/Entspannung. Upload oder FTP-Download.")

# ---------- Hilfsfunktionen ----------
PAT = re.compile(r"brainzz_(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})")

def parse_dt_from_path(path: str):
    m = PAT.search(path)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d--%H-%M-%S")
    except Exception:
        return None

def load_session_relatives(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    bands = ["Delta","Theta","Alpha","Beta","Gamma"]
    band_cols = {b:[c for c in df.columns if str(c).startswith(f"{b}_")] for b in bands}
    if not all(len(band_cols[b])>0 for b in bands):
        return None
    sums = {}
    for b in bands:
        sums[b.lower()] = df[band_cols[b]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    rel = pd.DataFrame(sums).replace([np.inf,-np.inf], np.nan).dropna()
    total = rel.sum(axis=1).replace(0, np.nan)
    rel = rel.div(total, axis=0).dropna()
    return rel

def build_session_table(root_dir: str)
