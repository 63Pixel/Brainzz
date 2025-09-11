#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-Auswertung Streamlit App - Version v4
Adds:
- Selection controls to limit which sessions from the package are analysed:
  * All (default)
  * Last 1 day, 3 days, 7 days, 14 days
  * Last N sessions
Behavior:
- Filtration uses timestamps parsed from filenames (pattern brainzz_YYYY-MM-DD--HH-MM-SS).
- If a file lacks a parsable timestamp it is ignored for time-based filters but can be included when "All" or "Last N sessions" is selected (Last N uses those with timestamps).
Save as app_streamlit_v4.py and deploy like before.
"""

import os, io, zipfile, glob, re, tempfile, shutil
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from ftplib import FTP
import plotly.express as px
import plotly.graph_objects as go

# Optional SFTP support
try:
    import paramiko
    HAS_PARAMIKO = True
except Exception:
    HAS_PARAMIKO = False

# Signal processing availability
try:
    from scipy.signal import iirnotch, butter, filtfilt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

st.set_page_config(page_title="EEG-Auswertung (v4)", layout="wide")
st.title("EEG-Auswertung — v4 (Session-Selection)")
st.caption("Wähle, wie viele Sessions aus dem Paket analysiert werden sollen.")

PAT = re.compile(r"brainzz_(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})")

def parse_dt_from_path(path: str):
    m = PAT.search(path)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d--%H-%M-%S")
    except Exception:
        return None

# (kept preprocessing and plotting helpers identical to v3 for brevity)
def notch_filter_signal(x, fs=250.0, f0=50.0, Q=30):
    if not HAS_SCIPY:
        return x
    b,a = iirnotch(f0, Q, fs)
    return filtfilt(b,a,x)

def bandpass_signal(x, fs=250.0, low=0.5, high=45.0, order=4):
    if not HAS_SCIPY:
        return x
    nyq = fs/2.0
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b,a,x)

def preprocess_csv_if_raw(csv_path, out_tmp_dir, fs=250.0):
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return csv_path, False
    band_prefixes = ("Delta_","Theta_","Alpha_","Beta_","Gamma_")
    non_band_cols = [c for c in df.columns if not str(c).startswith(band_prefixes)]
    numeric_cols = [c for c in non_band_cols if np.issubdtype(df[c].dtype, np.number)]
    if len(numeric_cols) < 2 or len(df) < 10 or not HAS_SCIPY:
        return csv_path, False
    proc = df.copy()
    for c in numeric_cols:
        try:
            sig = proc[c].astype(float).values
            sig = notch_filter_signal(sig, fs=fs)
            sig = bandpass_signal(sig, fs=fs)
            proc[c] = sig
        except Exception:
            continue
    out_path = os.path.join(out_tmp_dir, os.path.basename(csv_path).replace(".csv","_proc.csv"))
    proc.to_csv(out_path, index=False)
    return out_path, True

def load_session_relatives(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    bands = ["Delta","Theta","Alpha","Beta","Gamma"]
    band_cols = {}
    for b in bands:
        cols = [c for c in df.columns if str(c).startswith(f"{b}_")]
        if not cols and b in df.columns:
            cols = [b]
        band_cols[b] = cols
    if not all(len(band_cols[b])>0 for b in bands):
        return None
    sums = {}
    for b in bands:
        try:
            sums[b.lower()] = df[band_cols[b]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        except Exception:
            return None
    rel = pd.DataFrame(sums).replace([np.inf,-np.inf], np.nan).dropna()
    total = rel.sum(axis=1).replace(0, np.nan)
    rel = rel.div(total, axis=0).dropna()
    return rel

def build_session_table_from_list(csv_paths, tmpdir, fs=250.0, st_container=None):
    rows = []
    total = max(1, len(csv_paths))
    if st_container is not None:
        progress = st_container.progress(0)
        status_text = st_container.empty()
    for i, cp in enumerate(sorted(csv_paths), start=1):
        dt = parse_dt_from_path(cp) or parse_dt_from_path(os.path.dirname(cp))
        if dt is None:
            if st_container is not None:
                status_text.text(f"Skipping (no timestamp): {os.path.basename(cp)}")
            continue
        proc_path, did_proc = preprocess_csv_if_raw(cp, tmpdir, fs=fs)
        rel = load_session_relatives(proc_path)
        if rel is None or rel.empty:
            rel = load_session_relatives(cp)
            if rel is None or rel.empty:
                if st_container is not None:
                    status_text.text(f"Skipping (no band columns): {os.path.basename(cp)}")
                continue
        alpha = float(rel["alpha"].mean())
        beta  = float(rel["beta"].mean())
        theta = float(rel["theta"].mean())
        delta = float(rel["delta"].mean())
        gamma = float(rel["gamma"].mean())
        stress = float(beta/(alpha+1e-9))
        relax  = float(alpha/(beta+1e-9))
        rows.append({
            "datetime": dt,
            "alpha": alpha,
            "beta": beta,
            "theta": theta,
            "delta": delta,
            "gamma": gamma,
            "stress": stress,
            "relax": relax,
            "proc": did_proc,
            "source": os.path.basename(cp)
        })
        if st_container is not None:
            progress.progress(int(i/total*100))
            status_text.text(f"Processed {i}/{total}: {os.path.basename(cp)}")
    if st_container is not None:
        progress.empty()
        status_text.empty()
    df = pd.DataFrame(rows).dropna().sort_values("datetime").reset_index(drop=True)
    if not df.empty:
        df["date_str"] = df["datetime"].dt.strftime("%d-%m-%y %H:%M")
    return df

def try_compute_faa_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    left_keys = [k for k in df.columns if "alpha" in k.lower() and ("left" in k.lower() or "fp1" in k.lower() or "f3" in k.lower())]
    right_keys = [k for k in df.columns if "alpha" in k.lower() and ("right" in k.lower() or "fp2" in k.lower() or "f4" in k.lower())]
    if left_keys and right_keys:
        left = df[left_keys].apply(pd.to_numeric, errors="coerce").sum(axis=1).mean()
        right = df[right_keys].apply(pd.to_numeric, errors="coerce").sum(axis=1).mean()
        return np.log(right+1e-9) - np.log(left+1e-9)
    return None

# Plot helpers (Plotly)
def plot_single_session_interactive(df):
    vals = {
        "Stress": df["stress"].iloc[0],
        "Entspannung": df["relax"].iloc[0],
        "Delta": df["delta"].iloc[0],
        "Theta": df["theta"].iloc[0],
        "Alpha": df["alpha"].iloc[0],
        "Beta": df["beta"].iloc[0],
        "Gamma": df["gamma"].iloc[0],
    }
    data = pd.DataFrame({"Metrik": list(vals.keys()), "Wert": list(vals.values())})
    fig = px.bar(data, x="Metrik", y="Wert", color="Metrik", text="Wert", height=360)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", showlegend=False)
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    return fig

def plot_stress_relax(df, smooth:int=5, outlier_z=3.0):
    df = df.copy()
    df["stress_z"] = (df["stress"] - df["stress"].mean()) / (df["stress"].std(ddof=0) + 1e-9)
    df["relax_z"]  = (df["relax"]  - df["relax"].mean())  / (df["relax"].std(ddof=0)  + 1e-9)
    df["stress_trend"] = df["stress"].rolling(window=smooth, center=True, min_periods=1).mean()
    df["relax_trend"]  = df["relax"].rolling(window=smooth, center=True, min_periods=1).mean()
    long_df = df.melt(id_vars=["date_str"], value_vars=["stress","relax","stress_trend","relax_trend"],
                      var_name="Metrik", value_name="Wert")
    fig = px.line(long_df, x="date_str", y="Wert", color="Metrik", markers=True, height=360)
    fig.update_layout(xaxis=dict(type="category"))
    outl = df[(np.abs(df["stress_z"])>outlier_z) | (np.abs(df["relax_z"])>outlier_z)]
    if not outl.empty:
        for idx, row in outl.iterrows():
            fig.add_trace(go.Scatter(x=[row["date_str"]], y=[row["stress"]],
                                     mode="markers", marker_symbol="x", marker=dict(color="red", size=10),
                                     name="Outlier (Stress)", showlegend=False))
            fig.add_trace(go.Scatter(x=[row["date_str"]], y=[row["relax"]],
                                     mode="markers", marker_symbol="x", marker=dict(color="green", size=10),
                                     name="Outlier (Relax)", showlegend=False))
    fig.update_traces(hovertemplate="Datum: %{x}<br>%{y:.3f}")
    return fig

def plot_bands(df, smooth:int=5):
    d = df.copy()
    d["stresswave"] = d["beta"] + d["gamma"]
    d["relaxwave"]  = d["alpha"] + d["theta"]
    for c in ["delta","theta","alpha","beta","gamma","stresswave","relaxwave"]:
        d[f"{c}_trend"] = d[c].rolling(window=smooth, center=True, min_periods=1).mean()
    cols = ["delta_trend","theta_trend","alpha_trend","beta_trend","gamma_trend","stresswave_trend","relaxwave_trend"]
    long_df = d.melt(id_vars=["date_str"], value_vars=cols, var_name="Band", value_name="Wert")
    mapping = {
        "delta_trend":"Delta","theta_trend":"Theta","alpha_trend":"Alpha","beta_trend":"Beta","gamma_trend":"Gamma",
        "stresswave_trend":"Stress-Welle (Beta+Gamma)","relaxwave_trend":"Entspannungs-Welle (Alpha+Theta)"
    }
    long_df["Band"] = long_df["Band"].map(mapping)
    fig = px.line(long_df, x="date_str", y="Wert", color="Band", markers=True, height=380)
    fig.update_layout(xaxis=dict(type="category"), yaxis=dict(range=[0,1]))
    fig.update_traces(hovertemplate="Datum: %{x}<br>%{y:.3f}")
    return fig

# ---- UI ----
st.subheader("1) Datenquelle wählen")
mode = st.radio("Quelle", ["Datei-Upload (ZIP/Ordner als ZIP)", "FTP-Download", "SFTP (optional)"], horizontal=True)

workdir = tempfile.mkdtemp(prefix="eeg_works_")

if mode.startswith("Datei-Upload"):
    up = st.file_uploader("ZIP-Datei hochladen", type=["zip"])
    if up is not None:
        zbytes = up.read()
        with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
            zf.extractall(workdir)
        st.success("ZIP entpackt.")

elif mode.startswith("FTP"):
    st.info("FTP unverschlüsselt. Für SFTP wähle SFTP-Option.")
    host = st.text_input("FTP-Host", value="ftp.example.com")
    user = st.text_input("Benutzer", value="anonymous")
    pwd  = st.text_input("Passwort", value="", type="password")
    remote_dir = st.text_input("Remote-Pfad", value="/")
    pattern = st.text_input("Dateimuster (z. B. .zip)", value=".zip")
    go = st.button("Vom FTP laden")
    if go:
        try:
            ftp = FTP(host); ftp.login(user=user, passwd=pwd); ftp.cwd(remote_dir)
            names = ftp.nlst()
            targets = [n for n in names if pattern in n]
            for name in targets:
                loc = os.path.join(workdir, name)
                with open(loc, "wb") as f:
                    ftp.retrbinary(f"RETR {name}", f.write)
            st.success(f"{len(targets)} Datei(en) geladen.")
            ftp.quit()
        except Exception as e:
            st.error(f"FTP-Fehler: {e}")

elif mode.startswith("SFTP"):
    if not HAS_PARAMIKO:
        st.error("Paramiko nicht installiert. Installiere paramiko für SFTP-Unterstützung.")
    else:
        host = st.text_input("SFTP-Host", value="sftp.example.com")
        user = st.text_input("Benutzer", value="user")
        pwd  = st.text_input("Passwort", value="", type="password")
        remote_dir = st.text_input("Remote-Pfad", value="/")
        pattern = st.text_input("Dateimuster (z. B. .zip)", value=".zip")
        go = st.button("Vom SFTP laden")
        if go:
            try:
                transport = paramiko.Transport((host, 22))
                transport.connect(username=user, password=pwd)
                sftp = paramiko.SFTPClient.from_transport(transport)
                files = sftp.listdir(remote_dir)
                targets = [f for f in files if pattern in f]
                for name in targets:
                    remotepath = os.path.join(remote_dir, name)
                    localpath = os.path.join(workdir, name)
                    sftp.get(remotepath, localpath)
                sftp.close(); transport.close()
                st.success(f"{len(targets)} Datei(en) geladen via SFTP.")
            except Exception as e:
                st.error(f"SFTP-Fehler: {e}")

st.subheader("2) Parameter / QC")
smooth = st.slider("Glättungsfenster (Sessions)", min_value=3, max_value=11, value=5, step=2)
outlier_z = st.slider("Outlier z-Schwelle", min_value=1.5, max_value=5.0, value=3.0, step=0.5)
fs = st.number_input("Sampling-Rate für Preprocessing (Hz)", value=250.0, step=1.0)
do_preproc = st.checkbox("Versuche Notch+Bandpass-Preprocessing wenn Rohdaten vorhanden", value=True and HAS_SCIPY)
st.write("SciPy installiert:", HAS_SCIPY, "  Paramiko (SFTP):", HAS_PARAMIKO)

# --- New: Quick-check of file count and total size, option to limit newest N ---
csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*"), recursive=True)
                 if os.path.isfile(p) and p.lower().endswith(".csv")]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all) / (1024*1024) if n_csv>0 else 0.0
st.info(f"Gefundene CSVs: {n_csv}  —  Gesamtgröße: {total_mb:.1f} MB")

MAX_PREPROC_MB = 200
if total_mb > MAX_PREPROC_MB and do_preproc:
    st.warning(f"Preprocessing automatisch deaktiviert (Gesamt {total_mb:.0f} MB > {MAX_PREPROC_MB} MB).")
    do_preproc = False

# -- Session selection UI: by timeframe or last N sessions --
st.markdown("**Wähle, welche Sessions verarbeitet werden sollen**")
sel_mode = st.selectbox("Modus", ["Alle Sessions (default)", "Letzte Tage", "Letzte N Sessions"])
selected_csvs = csv_paths_all.copy()

if sel_mode == "Letzte Tage":
    days_opt = st.selectbox("Zeitraum", ["1 Tag", "3 Tage", "7 Tage", "14 Tage"], index=2)
    days_map = {"1 Tag":1, "3 Tage":3, "7 Tage":7, "14 Tage":14}
    days = days_map[days_opt]
    now = datetime.now()
    selected = []
    for p in csv_paths_all:
        dt = parse_dt_from_path(p)
        if dt is not None and dt >= (now - timedelta(days=days)):
            selected.append(p)
    selected_csvs = sorted(selected)
    st.write(f"{len(selected_csvs)} Sessions im Zeitraum der letzten {days} Tage gefunden.")
elif sel_mode == "Letzte N Sessions":
    max_n = len(csv_paths_all)
    default_n = min(30, max_n) if max_n>0 else 0
    n_sel = st.number_input("Anzahl neuester Sessions", min_value=1, max_value=max_n if max_n>0 else 1, value=default_n, step=1)
    # choose newest by parsed datetime; if parse fails, exclude
    parsed = [(p, parse_dt_from_path(p)) for p in csv_paths_all]
    parsed = [t for t in parsed if t[1] is not None]
    parsed_sorted = sorted(parsed, key=lambda x: x[1])
    selected_csvs = [p for p,_ in parsed_sorted[-int(n_sel):]]
    st.write(f"{len(selected_csvs)} Sessions ausgewählt (neueste {n_sel}).")
else:
    st.write(f"{len(selected_csvs)} Sessions (Alle)")

limit_sessions = st.checkbox("Beschränke Verarbeitung zusätzlich auf neueste N Sessions (beschleunigt)", value=False)
if limit_sessions and len(selected_csvs)>0:
    default_n2 = min(100, len(selected_csvs))
    sel_n2 = st.number_input(f"Wenn aktiv: verarbeite nur die neuesten N aus der Auswahl (max {len(selected_csvs)})", min_value=1, max_value=len(selected_csvs), value=default_n2, step=1)
    if sel_n2 < len(selected_csvs):
        # sort selected by parsed date and keep newest sel_n2
        parsed2 = [(p, parse_dt_from_path(p)) for p in selected_csvs]
        parsed2 = [t for t in parsed2 if t[1] is not None]
        parsed2_sorted = sorted(parsed2, key=lambda x: x[1])
        selected_csvs = [p for p,_ in parsed2_sorted[-int(sel_n2):]]
        st.write(f"Verarbeite jetzt {len(selected_csvs)} Sessions (neueste {sel_n2} aus Auswahl).")

if st.button("Auswertung starten"):
    # extract inner zips
    inner_zips = [os.path.join(workdir, f) for f in os.listdir(workdir) if f.lower().endswith(".zip")]
    for zp in inner_zips:
        name = os.path.splitext(os.path.basename(zp))[0]
        out = os.path.join(workdir, name); os.makedirs(out, exist_ok=True)
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(out)
        except zipfile.BadZipFile:
            pass

    # refresh selection after extraction: recompute csv_paths_all and intersect with selected_csvs by basename if needed
    csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*"), recursive=True)
                     if os.path.isfile(p) and p.lower().endswith(".csv")]
    # if selection was built before extraction, re-resolve by basename matching
    basenames = {os.path.basename(p): p for p in csv_paths_all}
    resolved_selected = []
    for p in selected_csvs:
        b = os.path.basename(p)
        if b in basenames:
            resolved_selected.append(basenames[b])
    # if selection empty after resolution, fall back to all csvs
    if not resolved_selected:
        resolved_selected = csv_paths_all

    # show final count
    st.info(f"Endgültige Anzahl zu verarbeitender Sessions: {len(resolved_selected)}")

    # temp dir for preprocessing outputs
    tmpdir = tempfile.mkdtemp(prefix="eeg_proc_")
    container = st.empty()
    df = build_session_table_from_list(resolved_selected, tmpdir, fs=fs if do_preproc else 0.0, st_container=container)
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    if df.empty:
        st.error("Keine gültigen Sessions gefunden. Prüfe ZIP-Inhalt.")
    else:
        faa_list = []
        for cp in resolved_selected:
            faa = try_compute_faa_from_csv(cp)
            if faa is not None:
                faa_list.append({"session": os.path.basename(cp), "faa": float(faa)})
        faa_df = pd.DataFrame(faa_list)

        if len(df) == 1:
            st.subheader("Einzel-Session Analyse")
            st.plotly_chart(plot_single_session_interactive(df), use_container_width=True)
            st.dataframe(df.round(4))
            if not faa_df.empty:
                st.write("FAA (geschätzt) für einzelne Sessions:"); st.dataframe(faa_df.round(4))
        else:
            st.subheader("Stress/Entspannung (ohne Lücken)")
            st.plotly_chart(plot_stress_relax(df, smooth=smooth, outlier_z=outlier_z), use_container_width=True)
            st.subheader("Bänder + Stress-/Entspannungswellen")
            st.plotly_chart(plot_bands(df, smooth=smooth), use_container_width=True)
            st.subheader("Session Übersicht (Tabelle)")
            st.dataframe(df.round(4))

        df_out = df.copy(); df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.download_button("Summary CSV herunterladen", data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="summary_indices.csv", mime="text/csv")
