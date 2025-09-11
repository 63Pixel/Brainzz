#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EEG-Auswertung Streamlit App

import os, io, re, glob, zipfile, tempfile, shutil
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from ftplib import FTP
import plotly.express as px
import plotly.graph_objects as go

# optional
try:
    import paramiko
    HAS_PARAMIKO = True
except Exception:
    HAS_PARAMIKO = False

try:
    from scipy.signal import iirnotch, butter, filtfilt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Dropbox
try:
    import dropbox
    from dropbox.files import FileMetadata, FolderMetadata
    from dropbox.exceptions import ApiError, AuthError
    HAS_DROPBOX = True
except Exception:
    HAS_DROPBOX = False

st.set_page_config(page_title="EEG-Auswertung", layout="wide")
st.title("EEG-Auswertung")
st.caption("ZIP/FTP/SFTP/Dropbox → Dateien wählen → Auswertung starten")

# ---------- Styles ----------
st.markdown("""
<style>
:root{ --expander-bg:#e9ecef; --expander-text:#0b3d91; --expander-open-bg:#28a745; --expander-open-text:#fff }
details > summary{ background:var(--expander-bg)!important; font-size:12px; font-weight:bold; color:var(--expander-text); padding:0.5em 1em; margin-bottom:1em; border-radius:0.5em; border:1px solid #ced4da; transition:all 0.2s ease-in-out }
details[open] > summary{ background:var(--expander-open-bg)!important; color:var(--expander-open-text); border:1px solid var(--expander-open-bg); }
details > summary::-webkit-details-marker{ display:none; }
div[data-baseweb="select"] { font-size:12px !important; }
div[role="listbox"]{ max-height:300px !important; font-size:12px !important; }
.ag-theme-streamlit-dark, .ag-theme-alpine{ --ag-font-size:12px; --ag-list-item-height:24px; }
</style>
""", unsafe_allow_html=True)


# --------- Helpers ----------
PAT = re.compile(r"brainzz_(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})")

def parse_dt_from_path(path: str):
    m = PAT.search(path)
    if not m: return None
    try: return datetime.strptime(m.group(1), "%Y-%m-%d--%H-%M-%S")
    except Exception: return None

def notch_filter_signal(x, fs=250.0, f0=50.0, Q=30):
    if not HAS_SCIPY: return x
    b,a = iirnotch(f0, Q, fs)
    return filtfilt(b,a,x)

def bandpass_signal(x, fs=250.0, low=0.5, high=45.0, order=4):
    if not HAS_SCIPY: return x
    nyq = fs/2.0
    b,a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b,a,x)

def preprocess_csv_if_raw(csv_path, out_tmp_dir, fs=250.0):
    try: df = pd.read_csv(csv_path, low_memory=False)
    except Exception: return csv_path, False
    band_prefixes = ("Delta_","Theta_","Alpha_","Beta_","Gamma_")
    non_band = [c for c in df.columns if not str(c).startswith(band_prefixes)]
    numeric = [c for c in non_band if np.issubdtype(df[c].dtype, np.number)]
    if len(numeric)<2 or len(df)<10 or not HAS_SCIPY: return csv_path, False
    proc = df.copy()
    for c in numeric:
        try:
            sig = proc[c].astype(float).values
            sig = notch_filter_signal(sig, fs=fs)
            sig = bandpass_signal(sig, fs=fs)
            proc[c] = sig
        except Exception: pass
    outp = os.path.join(out_tmp_dir, os.path.basename(csv_path).replace(".csv","_proc.csv"))
    proc.to_csv(outp, index=False)
    return outp, True

def load_session_relatives(csv_path):
    try: df = pd.read_csv(csv_path, low_memory=False)
    except Exception: return None
    bands = ["Delta","Theta","Alpha","Beta","Gamma"]
    cols = {b: [c for c in df.columns if str(c).startswith(f"{b}_")] or ([b] if b in df.columns else []) for b in bands}
    if not all(cols[b] for b in bands): return None
    sums = {}
    for b in bands:
        try: sums[b.lower()] = df[cols[b]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        except Exception: return None
    rel = pd.DataFrame(sums).replace([np.inf,-np.inf], np.nan).dropna()
    tot = rel.sum(axis=1).replace(0,np.nan)
    return rel.div(tot, axis=0).dropna()

def try_compute_faa_from_csv(csv_path):
    try: df = pd.read_csv(csv_path, low_memory=False)
    except Exception: return None
    L = [k for k in df.columns if "alpha" in k.lower() and any(tag in k.lower() for tag in ["left","fp1","f3"])]
    R = [k for k in df.columns if "alpha" in k.lower() and any(tag in k.lower() for tag in ["right","fp2","f4"])]
    if L and R:
        left = df[L].apply(pd.to_numeric, errors="coerce").sum(axis=1).mean()
        right= df[R].apply(pd.to_numeric, errors="coerce").sum(axis=1).mean()
        return np.log(right+1e-9)-np.log(left+1e-9)
    return None

def plot_single_session_interactive(df):
    vals = {"Stress": df["stress"].iloc[0], "Entspannung": df["relax"].iloc[0],
            "Delta": df["delta"].iloc[0], "Theta": df["theta"].iloc[0],
            "Alpha": df["alpha"].iloc[0], "Beta": df["beta"].iloc[0],
            "Gamma": df["gamma"].iloc[0]}
    data = pd.DataFrame({"Metrik": list(vals.keys()), "Wert": list(vals.values())})
    fig = px.bar(data, x="Metrik", y="Wert", color="Metrik", text="Wert", height=320)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", showlegend=False)
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10))
    return fig

def plot_stress_relax(df, smooth=5, outlier_z=3.0):
    d = df.copy()
    d["stress_trend"] = d["stress"].rolling(window=smooth, center=True, min_periods=1).mean()
    d["relax_trend"]  = d["relax"].rolling(window=smooth, center=True, min_periods=1).mean()
    long = d.melt(id_vars=["date_str"], value_vars=["stress","relax","stress_trend","relax_trend"],
                  var_name="Metrik", value_name="Wert")
    fig = px.line(long, x="date_str", y="Wert", color="Metrik", markers=True, height=360)
    fig.update_layout(xaxis=dict(type="category"))
    return fig

def plot_bands(df, smooth=5):
    d = df.copy()
    d["stresswave"] = d["beta"] + d["gamma"]
    d["relaxwave"]  = d["alpha"] + d["theta"]
    for c in ["delta","theta","alpha","beta","gamma","stresswave","relaxwave"]:
        d[f"{c}_trend"] = d[c].rolling(window=smooth, center=True, min_periods=1).mean()
    long = d.melt(id_vars=["date_str"],
                  value_vars=["delta_trend","theta_trend","alpha_trend","beta_trend","gamma_trend","stresswave_trend","relaxwave_trend"],
                  var_name="Band", value_name="Wert")
    mapn = {"delta_trend":"Delta","theta_trend":"Theta","alpha_trend":"Alpha","beta_trend":"Beta",
            "gamma_trend":"Gamma","stresswave_trend":"Stress-Welle (Beta+Gamma)","relaxwave_trend":"Entspannungs-Welle (Alpha+Theta)"}
    long["Band"] = long["Band"].map(mapn)
    fig = px.line(long, x="date_str", y="Wert", color="Band", markers=True, height=380)
    fig.update_layout(xaxis=dict(type="category"), yaxis=dict(range=[0,1]))
    return fig

def recursively_extract_archives(root_dir):
    changed = True
    while changed:
        changed = False
        archives = [p for p in glob.glob(os.path.join(root_dir,"**","*"), recursive=True)
                    if os.path.isfile(p) and p.lower().endswith((".zip",".sip"))]
        for arch in archives:
            if arch.endswith(".extracted"): continue
            try:
                target = os.path.join(os.path.dirname(arch), os.path.splitext(os.path.basename(arch))[0] + "_extracted")
                os.makedirs(target, exist_ok=True)
                with zipfile.ZipFile(arch,"r") as zf: zf.extractall(target)
                try: os.rename(arch, arch+".extracted")
                except Exception:
                    try: os.remove(arch)
                    except Exception: pass
                changed = True
            except zipfile.BadZipFile:
                continue
            except Exception:
                continue

def normalize_api_path(p: str) -> str:
    if p is None: return ""
    s = str(p).strip().replace("\\","/")
    if s in ("","/"): return ""
    return s if s.startswith("/") else "/" + s

def download_dropbox_file(dbx_token: str, remote_path: str, local_dir: str):
    dbx = dropbox.Dropbox(dbx_token, timeout=300)
    local_path = os.path.join(local_dir, os.path.basename(remote_path))
    try:
        md, res = dbx.files_download(remote_path)
        with open(local_path,"wb") as f: f.write(res.content)
    except Exception:
        dbx.files_download_to_file(local_path, remote_path)
    return local_path

def build_session_table_from_list(csv_paths, tmpdir, fs=250.0, st_container=None):
    rows=[]
    failed_files=[]
    total = max(1, len(csv_paths))
    if st_container is not None:
        progress = st_container.progress(0)
        status_text = st_container.empty()
    for i, cp in enumerate(sorted(csv_paths), start=1):
        dt = parse_dt_from_path(cp) or parse_dt_from_path(os.path.dirname(cp))
        if dt is None:
            failed_files.append({"source": os.path.basename(cp), "reason": "Ungültiger Zeitstempel im Namen."})
            if st_container is not None:
                st_container.warning(f"Datei '{os.path.basename(cp)}' übersprungen: Ungültiger Zeitstempel im Namen.")
            continue
        proc_path, did = preprocess_csv_if_raw(cp, tmpdir, fs=fs)
        rel = load_session_relatives(proc_path) or load_session_relatives(cp)
        if rel is None or rel.empty:
            failed_files.append({"source": os.path.basename(cp), "reason": "Keine gültigen Bandspalten gefunden."})
            if st_container is not None:
                st_container.warning(f"Datei '{os.path.basename(cp)}' übersprungen: Keine gültigen Bandspalten gefunden.")
            continue
        alpha, beta = float(rel["alpha"].mean()), float(rel["beta"].mean())
        theta, delta, gamma = float(rel["theta"].mean()), float(rel["delta"].mean()), float(rel["gamma"].mean())
        rows.append({
            "datetime": dt, "alpha":alpha,"beta":beta,"theta":theta,"delta":delta,"gamma":gamma,
            "stress": beta/(alpha+1e-9), "relax": alpha/(beta+1e-9), "source": os.path.basename(cp)
        })
        if st_container is not None:
            progress.progress(int(i/total*100))
            status_text.text(f"Processed {i}/{total}: {os.path.basename(cp)}")
    if st_container is not None:
        progress.empty(); status_text.empty()
    df = pd.DataFrame(rows)
    if not df.empty: df["date_str"] = df["datetime"].dt.strftime("%d-%m-%y %H:%M")
    return df, failed_files

# --------- UI: Quelle ---------
st.subheader("1) Datenquelle wählen")
mode = st.radio("Quelle", ["Datei-Upload (ZIP)", "FTP-Download", "SFTP (optional)", "Dropbox"], horizontal=True)
workdir = tempfile.mkdtemp(prefix="eeg_works_")
selected_paths = []

# Datei-Upload
if mode == "Datei-Upload (ZIP)":
    up = st.file_uploader("ZIP-Datei hochladen (CSV/SIP enthalten)", type=["zip"])
    if up is not None:
        try:
            with zipfile.ZipFile(io.BytesIO(up.read()),"r") as zf: zf.extractall(workdir)
            recursively_extract_archives(workdir)
            st.success("ZIP entpackt.")
        except Exception as e:
            st.error(f"ZIP-Fehler: {e}")

# FTP
elif mode == "FTP-Download":
    st.info("FTP unverschlüsselt. Für SFTP die SFTP-Option nutzen.")
    host = st.text_input("FTP-Host", "ftp.example.com")
    user = st.text_input("Benutzer", "anonymous")
    pwd  = st.text_input("Passwort", "", type="password")
    remote_dir = st.text_input("Remote-Pfad", "/")
    if st.button("Vom FTP laden"):
        try:
            ftp = FTP(host); ftp.login(user=user, passwd=pwd); ftp.cwd(remote_dir)
            names = ftp.nlst()
            for name in [n for n in names if n.lower().endswith(".zip")]:
                loc = os.path.join(workdir, name)
                with open(loc,"wb") as f: ftp.retrbinary("RETR "+name, f.write)
            ftp.quit()
            recursively_extract_archives(workdir)
            st.success("FTP: geladen und entpackt.")
        except Exception as e:
            st.error(f"FTP-Fehler: {e}")

# SFTP
elif mode == "SFTP (optional)":
    if not HAS_PARAMIKO:
        st.error("Paramiko nicht installiert.")
    else:
        host = st.text_input("SFTP-Host", "sftp.example.com")
        user = st.text_input("Benutzer", "user")
        pwd  = st.text_input("Passwort", "", type="password")
        remote_dir = st.text_input("Remote-Pfad", "/")
        if st.button("Vom SFTP laden"):
            try:
                transport = paramiko.Transport((host,22)); transport.connect(username=user, password=pwd)
                sftp = paramiko.SFTPClient.from_transport(transport)
                for name in [n for n in sftp.listdir(remote_dir) if n.lower().endswith(".zip")]:
                    sftp.get(os.path.join(remote_dir,name), os.path.join(workdir,name))
                sftp.close(); transport.close()
                recursively_extract_archives(workdir)
                st.success("SFTP: geladen und entpackt.")
            except Exception as e:
                st.error(f"SFTP-Fehler: {e}")

# Dropbox mit AgGrid
elif mode == "Dropbox":
    if not HAS_DROPBOX:
        st.error("Dropbox SDK fehlt. requirements.txt: dropbox")
    else:
        token = st.secrets.get("dropbox",{}).get("access_token") if "dropbox" in st.secrets else os.getenv("DROPBOX_TOKEN")
        configured_path = st.secrets.get("dropbox",{}).get("path") if "dropbox" in st.secrets else ""
        api_path = normalize_api_path(configured_path)
        if not token:
            st.error("Dropbox-Token fehlt.")
        else:
            st.markdown(f"**Dropbox-Ordner:** `{api_path if api_path else '(app-root)'}`")
            dbx = dropbox.Dropbox(token, timeout=300)

            # Listing
            entries=[]
            try:
                res = dbx.files_list_folder(api_path, recursive=False); entries = res.entries
            except AuthError:
                st.error("AuthError: Scopes fehlen (files.metadata.read, files.content.read). Neues Token.")
            except ApiError as e:
                st.error(f"Dropbox ApiError: {getattr(e,'error_summary',str(e))}")
            except Exception as e:
                st.error(f"Dropbox-Fehler: {e}")

            files = [e for e in entries if isinstance(e, FileMetadata)]
            files = [f for f in files if f.name.lower().endswith((".zip",".csv",".sip"))]
            files = sorted(files, key=lambda x: x.name.lower())

            # DataFrame für Anzeige: nur Beschreibung (Datum · Größe)
            rows=[]
            for f in files:
                dt = parse_dt_from_path(f.path_display)
                dt_str = dt.strftime("%d-%m-%y %H:%M") if dt else ""
                size_kb = int(getattr(f,"size",0)/1024) if getattr(f,"size",None) is not None else None
                size_str = f"{size_kb} KB" if size_kb is not None else ""
                desc = (dt_str + " · " + size_str).strip(" ·")
                rows.append({"description": desc, "path_lower": f.path_lower, "path_display": f.path_display,
                             "dt": dt_str, "size_kb": size_kb})
            df_files = pd.DataFrame(rows)

            # AgGrid (Checkbox-Selection). Fallback = native Checkbox-Liste.
            selected_paths = []
            try:
                from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
                if df_files.empty:
                    st.info("0 Datei(en) im Ordner.")
                else:
                    gb = GridOptionsBuilder.from_dataframe(df_files)
                    # Nur Beschreibung zeigen
                    gb.configure_column("description", header_name="Datum · Größe",
                                         checkboxSelection=True, headerCheckboxSelection=True,
                                         headerCheckboxSelectionFilteredOnly=True,
                                         sortable=True, filter=True, resizable=True)
                    # Verstecken
                    for col in ["path_lower","path_display","dt","size_kb"]:
                        gb.configure_column(col, hide=True)
                    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
                    gb.configure_grid_options(
                        rowSelection="multiple",
                        suppressRowClickSelection=True,
                        rowMultiSelectWithClick=True,
                        pagination=True, paginationAutoPageSize=False, paginationPageSize=30,
                        domLayout='normal'
                    )
                    grid = AgGrid(
                        df_files[["description","path_lower","path_display","dt","size_kb"]],
                        gridOptions=gb.build(),
                        height=360,
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        enable_enterprise_modules=False,
                        allow_unsafe_jscode=False,
                        theme="streamlit"
                    )
                    selected = grid.get("selected_rows", [])
                    selected_paths = [r["path_lower"] for r in selected] if selected else []
            except Exception as e:
                st.warning("AgGrid nicht verfügbar. Fallback auf einfache Checkbox-Liste.")
                if df_files.empty:
                    st.info("0 Datei(en) im Ordner.")
                else:
                    st.session_state.setdefault("fb_sel", set())
                    for i,row in df_files.iterrows():
                        key = f"fb_{i}"
                        checked = st.checkbox(row["description"], key=key, value=(key in st.session_state["fb_sel"]))
                        if checked: st.session_state["fb_sel"].add(key)
                        else: st.session_state["fb_sel"].discard(key)
                    # Mappe Auswahl auf Pfade
                    for i,row in df_files.iterrows():
                        if f"fb_{i}" in st.session_state["fb_sel"]:
                            selected_paths.append(row["path_lower"])

            # Download-Action
            dl_col1, dl_col2 = st.columns([1,3])
            with dl_col1:
                if st.button("Herunterladen (Auswahl)"):
                    if not selected_paths:
                        st.warning("Keine Datei ausgewählt.")
                    else:
                        downloaded=[]
                        for remote in selected_paths:
                            try:
                                lp = download_dropbox_file(token, remote, workdir)
                                downloaded.append(lp)
                            except Exception as e:
                                st.error(f"Download fehlgeschlagen: {remote} — {e}")
                        if downloaded:
                            recursively_extract_archives(workdir)
                            st.success(f"{len(downloaded)} Datei(en) heruntergeladen und extrahiert.")
            
# --------- Parameter / QC (behalten, kompakt) ----------
st.subheader("2) Parameter / QC")
with st.expander("Hilfe zu Parametern", expanded=False):
    st.markdown("""
**Glättungsfenster**: Sessions für Trend-Glättung (3–7).  
**Outlier z-Schwelle**: 2.5–3.5 üblich.  
**Sampling-Rate**: Nur für Rohdaten-Preprocessing nötig.
""")
smooth = st.slider("Glättungsfenster (Sessions)", 3, 11, 5, 2)
outlier_z = st.slider("Outlier z-Schwelle", 1.5, 5.0, 3.0, 0.5)
fs = st.number_input("Sampling-Rate für Preprocessing (Hz)", value=250.0, step=1.0)
do_preproc = st.checkbox("Preprocessing (Notch+Bandpass), falls Rohdaten", value=(True and HAS_SCIPY))

# --------- CSV-Suche im Arbeitsverzeichnis ---------
csv_paths_all = [p for p in glob.glob(os.path.join(workdir,"**","*"), recursive=True)
                 if os.path.isfile(p) and p.lower().endswith(".csv")]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all) / (1024*1024) if n_csv>0 else 0.0
st.info(f"Gefundene CSVs: {n_csv}  —  Gesamtgröße: {total_mb:.1f} MB")

# --------- Auswertung: jetzt mit der richtigen Dateiauswahl ---------
if st.button("Auswertung starten"):
    recursively_extract_archives(workdir)
    
    # Hier wird die Auswahl aus der Dropbox-Tabelle verarbeitet.
    # Wenn im Dropbox-Modus Dateien ausgewählt wurden, werden nur diese analysiert.
    if mode == "Dropbox" and selected_paths:
        selected_csvs = [os.path.join(workdir, os.path.basename(path)) for path in selected_paths]
    else:
        # Andernfalls, oder wenn keine Dateien ausgewählt wurden, werden alle gefundenen CSVs verarbeitet.
        selected_csvs = [p for p in glob.glob(os.path.join(workdir,"**","*"), recursive=True)
                         if os.path.isfile(p) and p.lower().endswith(".csv")]
        
    st.info(f"Endgültige Anzahl zu verarbeitender Sessions: {len(selected_csvs)}")

    if not selected_csvs:
        st.error("Keine CSVs gefunden. Lade Dateien und versuche es erneut.")
    else:
        tmpdir = tempfile.mkdtemp(prefix="eeg_proc_")
        container = st.empty()
        df, failed_files = build_session_table_from_list(selected_csvs, tmpdir, fs=fs if do_preproc else 0.0, st_container=container)
        try: shutil.rmtree(tmpdir)
        except Exception: pass

        # **Neue Logik:**
        if not df.empty:
            faa_list = []
            for cp in selected_csvs:
                faa = try_compute_faa_from_csv(cp)
                if faa is not None:
                    faa_list.append({"session": os.path.basename(cp), "faa": float(faa)})
            faa_df = pd.DataFrame(faa_list)
            
            if len(df)==1:
                st.subheader("Einzel-Session")
                st.plotly_chart(plot_single_session_interactive(df), use_container_width=True)
                st.dataframe(df.round(4))
                if not faa_df.empty:
                    st.write("FAA (geschätzt) für einzelne Sessions:")
                    st.dataframe(faa_df.round(4))
            else:
                st.subheader("Stress/Entspannung")
                st.plotly_chart(plot_stress_relax(df, smooth=smooth, outlier_z=outlier_z), use_container_width=True)
                st.subheader("Bänder + Wellen")
                st.plotly_chart(plot_bands(df, smooth=smooth), use_container_width=True)
                st.subheader("Tabelle")
                st.dataframe(df.round(4))

            df_out = df.copy()
            df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.download_button("Summary CSV herunterladen",
                               data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name="summary_indices.csv", mime="text/csv")
        
        # **Ausgabe der fehlgeschlagenen Dateien:**
        if failed_files:
            st.subheader("Folgende Dateien wurden übersprungen:")
            for f in failed_files:
                st.warning(f"**{f['source']}**: {f['reason']}")
