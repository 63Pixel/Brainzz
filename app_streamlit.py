#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-Auswertung Streamlit App
- Dropbox: kein Verzeichniswechsel, nur Dateien im konfigurierten Ordner
- Checkbox "Alle auswählen" + Multiselect
- ZIP/SIP werden nach Download rekursiv extrahiert
- Analyse (Stress/Entspannung, Bänder, FAA)
"""

import os
import io
import zipfile
import glob
import re
import tempfile
import shutil
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

# ---------- Styles ----------
st.markdown("""
<style>
:root{ --expander-bg:#e9ecef; --expander-text:#0b3d91; --expander-open-bg:#28a745; --expander-open-text:#fff }
details > summary{ background:var(--expander-bg)!important; color:var(--expander-text)!important; padding:8px 12px; border-radius:8px; cursor:pointer }
details[open] > summary{ background:var(--expander-open-bg)!important; color:var(--expander-open-text)!important }
</style>
""", unsafe_allow_html=True)

st.title("EEG-Auswertung")
st.caption("ZIP/FTP/SFTP/Dropbox → Dateien wählen → Auswertung starten")

# ---------- Helpers ----------
PAT = re.compile(r"brainzz_(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})")

def parse_dt_from_path(path: str):
    m = PAT.search(path)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d--%H-%M-%S")
    except Exception:
        return None

def notch_filter_signal(x, fs=250.0, f0=50.0, Q=30):
    if not HAS_SCIPY:
        return x
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x)

def bandpass_signal(x, fs=250.0, low=0.5, high=45.0, order=4):
    if not HAS_SCIPY:
        return x
    nyq = fs/2.0
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

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

def load_session_relatives(csv_path):
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

def plot_single_session_interactive(df):
    vals = {"Stress": df["stress"].iloc[0], "Entspannung": df["relax"].iloc[0],
            "Delta": df["delta"].iloc[0], "Theta": df["theta"].iloc[0],
            "Alpha": df["alpha"].iloc[0], "Beta": df["beta"].iloc[0],
            "Gamma": df["gamma"].iloc[0]}
    data = pd.DataFrame({"Metrik": list(vals.keys()), "Wert": list(vals.values())})
    fig = px.bar(data, x="Metrik", y="Wert", color="Metrik", text="Wert", height=360)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", showlegend=False)
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    return fig

def plot_stress_relax(df, smooth=5, outlier_z=3.0):
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
        for _, row in outl.iterrows():
            fig.add_trace(go.Scatter(x=[row["date_str"]], y=[row["stress"]],
                                     mode="markers", marker_symbol="x",
                                     marker=dict(color="red", size=10),
                                     name="Outlier (Stress)", showlegend=False))
            fig.add_trace(go.Scatter(x=[row["date_str"]], y=[row["relax"]],
                                     mode="markers", marker_symbol="x",
                                     marker=dict(color="green", size=10),
                                     name="Outlier (Relax)", showlegend=False))
    fig.update_traces(hovertemplate="Datum: %{x}<br>%{y:.3f}")
    return fig

def plot_bands(df, smooth=5):
    d = df.copy()
    d["stresswave"] = d["beta"] + d["gamma"]
    d["relaxwave"]  = d["alpha"] + d["theta"]
    for c in ["delta","theta","alpha","beta","gamma","stresswave","relaxwave"]:
        d[f"{c}_trend"] = d[c].rolling(window=smooth, center=True, min_periods=1).mean()
    cols = ["delta_trend","theta_trend","alpha_trend","beta_trend","gamma_trend","stresswave_trend","relaxwave_trend"]
    long_df = d.melt(id_vars=["date_str"], value_vars=cols, var_name="Band", value_name="Wert")
    mapping = {"delta_trend":"Delta","theta_trend":"Theta","alpha_trend":"Alpha","beta_trend":"Beta",
               "gamma_trend":"Gamma","stresswave_trend":"Stress-Welle (Beta+Gamma)",
               "relaxwave_trend":"Entspannungs-Welle (Alpha+Theta)"}
    long_df["Band"] = long_df["Band"].map(mapping)
    fig = px.line(long_df, x="date_str", y="Wert", color="Band", markers=True, height=380)
    fig.update_layout(xaxis=dict(type="category"), yaxis=dict(range=[0,1]))
    fig.update_traces(hovertemplate="Datum: %{x}<br>%{y:.3f}")
    return fig

def recursively_extract_archives(root_dir):
    changed = True
    while changed:
        changed = False
        archives = [p for p in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
                    if os.path.isfile(p) and p.lower().endswith((".zip", ".sip"))]
        for arch in archives:
            if arch.endswith(".extracted"):
                continue
            try:
                target = os.path.join(os.path.dirname(arch), os.path.splitext(os.path.basename(arch))[0] + "_extracted")
                os.makedirs(target, exist_ok=True)
                with zipfile.ZipFile(arch, "r") as zf:
                    zf.extractall(target)
                try:
                    os.rename(arch, arch + ".extracted")
                except Exception:
                    try:
                        os.remove(arch)
                    except Exception:
                        pass
                changed = True
            except zipfile.BadZipFile:
                continue
            except Exception:
                continue

# ---------- Dropbox helpers ----------
def normalize_api_path(p: str) -> str:
    """'' -> app-root; sonst führender '/'."""
    if p is None:
        return ""
    s = str(p).strip().replace("\\", "/")
    if s in ("", "/"):
        return ""
    return s if s.startswith("/") else "/" + s

def list_folder(dbx, api_path: str):
    """Nur direkte Einträge des konfigurierten Ordners."""
    try:
        res = dbx.files_list_folder(api_path, recursive=False)
        entries = res.entries
    except AuthError:
        st.error("AuthError: Token ohne nötige Scopes. Aktiviere files.metadata.read + files.content.read und erzeuge ein neues Token.")
        return [], []
    except ApiError as e:
        try:
            if hasattr(e.error, "get_path") and e.error.get_path() and e.error.get_path().is_not_found():
                st.error("Pfad nicht gefunden. Prüfe st.secrets['dropbox']['path'] und App-Typ (App folder vs Full Dropbox).")
            else:
                st.error(f"Dropbox ApiError: {e.error_summary}")
        except Exception:
            st.error(f"Dropbox ApiError: {e}")
        return [], []
    except Exception as e:
        st.error(f"Dropbox-Fehler: {e}")
        return [], []
    folders = [e for e in entries if isinstance(e, FolderMetadata)]
    files   = [e for e in entries if isinstance(e, FileMetadata)]
    return folders, files

def download_dropbox_file(dbx_token: str, remote_path: str, local_dir: str):
    dbx = dropbox.Dropbox(dbx_token, timeout=300)
    local_path = os.path.join(local_dir, os.path.basename(remote_path))
    try:
        md, res = dbx.files_download(remote_path)
        with open(local_path, "wb") as f:
            f.write(res.content)
    except Exception:
        dbx.files_download_to_file(local_path, remote_path)
    return local_path

# ---------- Build table ----------
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
            "alpha": alpha, "beta": beta, "theta": theta, "delta": delta, "gamma": gamma,
            "stress": stress, "relax": relax,
            "proc": did_proc, "source": os.path.basename(cp)
        })
        if st_container is not None:
            progress.progress(int(i/total*100))
            status_text.text(f"Processed {i}/{total}: {os.path.basename(cp)}")
    if st_container is not None:
        progress.empty(); status_text.empty()
    df = pd.DataFrame(rows).dropna().sort_values("datetime").reset_index(drop=True)
    if not df.empty:
        df["date_str"] = df["datetime"].dt.strftime("%d-%m-%y %H:%M")
    return df

# ---------- UI ----------
st.subheader("1) Datenquelle wählen")
mode = st.radio("Quelle", ["Datei-Upload (ZIP)", "FTP-Download", "SFTP (optional)", "Dropbox"], horizontal=True)

workdir = tempfile.mkdtemp(prefix="eeg_works_")

# Upload
if mode == "Datei-Upload (ZIP)":
    up = st.file_uploader("ZIP-Datei hochladen (Paket mit CSVs/SIPs)", type=["zip"])
    if up is not None:
        zbytes = up.read()
        try:
            with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
                zf.extractall(workdir)
            recursively_extract_archives(workdir)
            st.success("ZIP entpackt und verschachtelte Archive extrahiert.")
        except Exception as e:
            st.error(f"ZIP konnte nicht entpackt werden: {e}")

# FTP
elif mode == "FTP-Download":
    st.info("FTP unverschlüsselt. Für SFTP wähle SFTP-Option.")
    host = st.text_input("FTP-Host", value="ftp.example.com")
    user = st.text_input("Benutzer", value="anonymous")
    pwd  = st.text_input("Passwort", value="", type="password")
    remote_dir = st.text_input("Remote-Pfad", value="/")
    pattern = st.text_input("Dateimuster (z. B. .zip)", value=".zip")
    if st.button("Vom FTP laden"):
        try:
            ftp = FTP(host); ftp.login(user=user, passwd=pwd); ftp.cwd(remote_dir)
            names = ftp.nlst()
            targets = [n for n in names if pattern in n]
            if not targets:
                st.warning("Keine passenden Dateien auf FTP gefunden.")
            for name in targets:
                loc = os.path.join(workdir, name)
                with open(loc, "wb") as f:
                    ftp.retrbinary("RETR " + name, f.write)
            recursively_extract_archives(workdir)
            st.success(f"{len(targets)} Datei(en) geladen und Archive extrahiert.")
            ftp.quit()
        except Exception as e:
            st.error(f"FTP-Fehler: {e}")

# SFTP
elif mode == "SFTP (optional)":
    if not HAS_PARAMIKO:
        st.error("Paramiko nicht installiert. SFTP nicht verfügbar.")
    else:
        host = st.text_input("SFTP-Host", value="sftp.example.com")
        user = st.text_input("Benutzer", value="user")
        pwd  = st.text_input("Passwort", value="", type="password")
        remote_dir = st.text_input("Remote-Pfad", value="/")
        pattern = st.text_input("Dateimuster (z. B. .zip)", value=".zip")
        if st.button("Vom SFTP laden"):
            try:
                transport = paramiko.Transport((host, 22))
                transport.connect(username=user, password=pwd)
                sftp = paramiko.SFTPClient.from_transport(transport)
                files = sftp.listdir(remote_dir)
                targets = [f for f in files if pattern in f]
                if not targets:
                    st.warning("Keine passenden Dateien auf SFTP gefunden.")
                for name in targets:
                    remotepath = os.path.join(remote_dir, name)
                    localpath = os.path.join(workdir, name)
                    sftp.get(remotepath, localpath)
                sftp.close(); transport.close()
                recursively_extract_archives(workdir)
                st.success(f"{len(targets)} Datei(en) geladen via SFTP und Archive extrahiert.")
            except Exception as e:
                st.error(f"SFTP-Fehler: {e}")

# ---------------- Dropbox: Kachel-Grid, Pagination, Checkboxen (ersetze bisherigen Dropbox-Block) ----------------
# ---------------- Dropbox: Spalten-/Tabellenansicht (AgGrid) ----------------
elif mode == "Dropbox":
    if not HAS_DROPBOX:
        st.error("Dropbox SDK fehlt. Füge 'dropbox' zu requirements.txt hinzu.")
    else:
        token = st.secrets.get("dropbox", {}).get("access_token") if "dropbox" in st.secrets else os.getenv("DROPBOX_TOKEN")
        configured_path = st.secrets.get("dropbox", {}).get("path") if "dropbox" in st.secrets else ""
        if configured_path is None:
            configured_path = ""
        api_path = normalize_api_path(configured_path)

        if not token:
            st.error("Dropbox-Token fehlt (st.secrets['dropbox']['access_token'] oder DROPBOX_TOKEN).")
        else:
            st.markdown(f"**Dropbox-Ordner:** `{api_path if api_path else '(app-root)'}`")
            dbx = dropbox.Dropbox(token, timeout=300)

            # Listing (nur direkte Einträge)
            try:
                res = dbx.files_list_folder(api_path, recursive=False)
                entries = res.entries
            except dropbox.exceptions.AuthError:
                st.error("AuthError: Token ohne nötige Scopes. Erzeuge neues Token mit files.metadata.read + files.content.read.")
                entries = []
            except dropbox.exceptions.ApiError as e:
                st.error(f"Dropbox ApiError: {getattr(e,'error_summary',str(e))}")
                entries = []
            except Exception as e:
                st.error(f"Dropbox-Fehler: {e}")
                entries = []

            files = [e for e in entries if isinstance(e, dropbox.files.FileMetadata)]
            visible_files = [f for f in files if f.name.lower().endswith((".zip", ".csv", ".sip"))]
            visible_files = sorted(visible_files, key=lambda x: x.name.lower())

            try:
                from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
                if df_files.empty:
                    st.info("0 Datei(en) im Ordner (nur direkte Einträge, keine Unterordner).")
                else:
                    gb = GridOptionsBuilder.from_dataframe(df_files)
                    gb.configure_column("description", header_name="Datum · Größe", sortable=True, filter=True, resizable=True)
                    gb.configure_column("dt", header_name="Datum", hide=True)
                    gb.configure_column("size_kb", header_name="SizeKB", hide=True)
                    gb.configure_column("path_display", header_name="Name", hide=True)
                    gb.configure_column("path_lower", header_name="remote", hide=True)
                    gb.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=[])
                    gb.configure_grid_options(domLayout='normal')
                    gridOptions = gb.build()

                    grid_response = AgGrid(
                        df_files[["description","path_display","path_lower","dt","size_kb"]],
                        gridOptions=gridOptions,
                        height=300,
                        update_on=[GridUpdateMode.SELECTION_CHANGED],
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        fit_columns_on_grid_load=True,
                        enable_enterprise_modules=False,
                        allow_unsafe_jscode=False,
                    )

                    selected = grid_response.get("selected_rows", [])
                    selected_paths = [r.get("path_lower") for r in selected] if selected else []

                    st.markdown("---")
                    c1, c2, c3 = st.columns([3,1,1])
                    with c1:
                        st.markdown(f"Ausgewählt: **{len(selected_paths)}**")
                    with c2:
                        if st.button("Herunterladen ausgewählter Dateien"):
                            if not selected_paths:
                                st.warning("Keine Datei ausgewählt.")
                            else:
                                downloaded = []
                                for remote in selected_paths:
                                    try:
                                        lp = download_dropbox_file(token, remote, workdir)
                                        downloaded.append(lp)
                                    except Exception as e:
                                        st.error(f"Download fehlgeschlagen: {remote} — {e}")
                                if downloaded:
                                    recursively_extract_archives(workdir)
                                    st.success(f"{len(downloaded)} Datei(en) heruntergeladen und extrahiert.")
                    with c3:
                        if st.button("Alle abwählen"):
                            st.experimental_rerun()

            except Exception:
                st.error("Fehler beim Laden von streamlit-aggrid. Bitte stelle sicher, dass die Bibliothek korrekt installiert ist.")


# ---------- Parameter / QC ----------
st.subheader("2) Parameter / QC")
with st.expander("Hilfe zu Parametern", expanded=False):
    st.markdown("""
**Glättungsfenster (Sessions)** - Sessions für Trend-Glättung. Empfehlung: 3–7.

**Outlier z-Schwelle** - Markiert starke Ausreißer. Empfehlung: 2.5–3.5.

**Sampling-Rate (Hz)** - Nur für Rohdaten-Preprocessing nötig (Notch/Bandpass).
""")

smooth = st.slider("Glättungsfenster (Sessions)", min_value=3, max_value=11, value=5, step=2)
outlier_z = st.slider("Outlier z-Schwelle", min_value=1.5, max_value=5.0, value=3.0, step=0.5)
fs = st.number_input("Sampling-Rate für Preprocessing (Hz)", value=250.0, step=1.0)
do_preproc = st.checkbox("Preprocessing (Notch+Bandpass), falls Rohdaten", value=(True and HAS_SCIPY))

# ---------- CSV sammeln ----------
csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*"), recursive=True)
                 if os.path.isfile(p) and p.lower().endswith(".csv")]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all) / (1024*1024) if n_csv>0 else 0.0
st.info(f"Gefundene CSVs: {n_csv}  —  Gesamtgröße: {total_mb:.1f} MB")

# ---------- Session Auswahl aus AgGrid-Tabelle übernehmen ----------
if mode == "Dropbox":
    selected_csvs = [os.path.join(workdir, os.path.basename(path)) for path in selected_paths]
else:
    selected_csvs = csv_paths_all

# ---------- Auswertung ----------
if st.button("Auswertung starten"):
    recursively_extract_archives(workdir)

    csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*"), recursive=True)
                     if os.path.isfile(p) and p.lower().endswith(".csv")]
    csv_paths_all = sorted(csv_paths_all)
    basenames_map = {os.path.basename(p): p for p in csv_paths_all}

    resolved_selected = []
    for p in selected_csvs:
        b = os.path.basename(p)
        if b in basenames_map:
            resolved_selected.append(basenames_map[b])
    if not resolved_selected:
        resolved_selected = csv_paths_all

    st.info(f"Endgültige Anzahl zu verarbeitender Sessions: {len(resolved_selected)}")

    if len(resolved_selected) == 0:
        st.error("Keine CSVs zum Verarbeiten gefunden.")
    else:
        tmpdir = tempfile.mkdtemp(prefix="eeg_proc_")
        container = st.empty()
        df = build_session_table_from_list(resolved_selected, tmpdir, fs=fs if do_preproc else 0.0, st_container=container)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

        if df.empty:
            st.error("Keine gültigen Sessions (mit Bandspalten) gefunden.")
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
                    st.write("FAA (geschätzt) für einzelne Sessions:")
                    st.dataframe(faa_df.round(4))
            else:
                st.subheader("Stress/Entspannung (ohne Lücken)")
                st.plotly_chart(plot_stress_relax(df, smooth=smooth, outlier_z=outlier_z), use_container_width=True)
                st.subheader("Bänder + Stress-/Entspannungswellen")
                st.plotly_chart(plot_bands(df, smooth=smooth), use_container_width=True)
                st.subheader("Session Übersicht (Tabelle)")
                st.dataframe(df.round(4))

            df_out = df.copy()
            df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.download_button("Summary CSV herunterladen",
                               data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name="summary_indices.csv", mime="text/csv")
