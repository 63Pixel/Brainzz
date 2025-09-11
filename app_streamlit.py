#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG-Auswertung Streamlit App
- Rekursive Extraktion (.zip/.sip)
- Kategorie-X-Achsen Plotly
- Hilfe-Expander standardmäßig eingeklappt
- "Alle Sessions" Ansicht: Hervorhebung letzter Tage / neueste N (Badges)
- Keine Debug-Ausgaben / keine SciPy/Paramiko-Statuszeile
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

# Optional libs
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

st.set_page_config(page_title="EEG-Auswertung", layout="wide")

# --- CSS für Badges ---
st.markdown("""
<style>
.badge{display:inline-block;padding:4px 8px;border-radius:8px;color:#fff;font-size:12px;margin-right:8px}
.badge-recent{background:#ff8c00}
.badge-newest{background:#28a745}
.badge-both{background:#6f42c1}
.session-row{padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.03)}
</style>
""", unsafe_allow_html=True)

st.title("EEG-Auswertung")
st.caption("Interaktive Auswertung. ZIP/FTP/SFTP hochladen, Sessions auswählen, Analyse starten.")

# Regex to parse filename timestamps like: brainzz_2025-09-05--09-47-44
PAT = re.compile(r"brainzz_(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})")

def parse_dt_from_path(path):
    m = PAT.search(path)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d--%H-%M-%S")
    except Exception:
        return None

# --- optional signal processing helpers (best-effort)
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

# --- Plot helpers (Plotly) ---
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
        for idx, row in outl.iterrows():
            fig.add_trace(go.Scatter(x=[row["date_str"]], y=[row["stress"]],
                                     mode="markers", marker_symbol="x", marker=dict(color="red", size=10),
                                     name="Outlier (Stress)", showlegend=False))
            fig.add_trace(go.Scatter(x=[row["date_str"]], y=[row["relax"]],
                                     mode="markers", marker_symbol="x", marker=dict(color="green", size=10),
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
    mapping = {
        "delta_trend":"Delta","theta_trend":"Theta","alpha_trend":"Alpha","beta_trend":"Beta","gamma_trend":"Gamma",
        "stresswave_trend":"Stress-Welle (Beta+Gamma)","relaxwave_trend":"Entspannungs-Welle (Alpha+Theta)"
    }
    long_df["Band"] = long_df["Band"].map(mapping)
    fig = px.line(long_df, x="date_str", y="Wert", color="Band", markers=True, height=380)
    fig.update_layout(xaxis=dict(type="category"), yaxis=dict(range=[0,1]))
    fig.update_traces(hovertemplate="Datum: %{x}<br>%{y:.3f}")
    return fig

# ---------------- helper: recursive extraction of nested archives ----------------
def recursively_extract_archives(root_dir):
    changed = True
    while changed:
        changed = False
        archives = [p for p in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
                    if os.path.isfile(p) and p.lower().endswith(('.zip', '.sip'))]
        for arch in archives:
            if arch.endswith('.extracted'):
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

# ---------------- UI ----------------
st.subheader("1) Datenquelle wählen")
mode = st.radio("Quelle", ["Datei-Upload (ZIP)", "FTP-Download", "SFTP (optional)"], horizontal=True)

workdir = tempfile.mkdtemp(prefix="eeg_works_")

if mode.startswith("Datei-Upload"):
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

elif mode.startswith("FTP"):
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

elif mode.startswith("SFTP"):
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

# ---------- Parameters / QC ----------
st.subheader("2) Parameter / QC")
with st.expander("Hilfe zu Parametern", expanded=False):
    st.markdown("""
**Glättungsfenster (Sessions)**  
- Anzahl Sessions für die Trend-Glättung. Größer = glatter, kleiner = detailreicher. Empfehlung: 3–7.

**Outlier z-Schwelle**  
- Markiert Messwerte mit hoher Abweichung. Empfehlung: 2.5–3.5.

**Sampling-Rate (Hz)**  
- Nur für Rohdaten-Preprocessing nötig (Notch/Bandpass). Häufig 250 Hz.

**Preprocessing**  
- Notch entfernt Netzstörungen (50/60 Hz). Bandpass begrenzt auf 0.5–45 Hz.
- Nur aktivieren wenn CSV Rohzeitreihen enthält. SciPy benötigt.

**Was sind "Endsessions"?**  
- "Endsessions" = die neuesten N Sessions. Das sind die zuletzt aufgezeichneten Messungen nach Timestamp.
""")

smooth = st.slider("Glättungsfenster (Sessions)", min_value=3, max_value=11, value=5, step=2)
outlier_z = st.slider("Outlier z-Schwelle", min_value=1.5, max_value=5.0, value=3.0, step=0.5)
fs = st.number_input("Sampling-Rate für Preprocessing (Hz)", value=250.0, step=1.0)
do_preproc = st.checkbox("Versuche Notch+Bandpass-Preprocessing wenn Rohdaten vorhanden", value=(True and HAS_SCIPY))

# --- gather CSVs robustly (case-insensitive) ---
csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*"), recursive=True)
                 if os.path.isfile(p) and p.lower().endswith(".csv")]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all) / (1024*1024) if n_csv>0 else 0.0

st.info(f"Gefundene CSVs: {n_csv}  —  Gesamtgröße: {total_mb:.1f} MB")
MAX_PREPROC_MB = 200
if total_mb > MAX_PREPROC_MB and do_preproc:
    st.warning(f"Preprocessing automatisch deaktiviert (Gesamt {total_mb:.0f} MB > {MAX_PREPROC_MB} MB).")
    do_preproc = False

# ---------------- Session selection UI ----------------
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
    parsed = [(p, parse_dt_from_path(p)) for p in csv_paths_all]
    parsed = [t for t in parsed if t[1] is not None]
    if not parsed:
        st.warning("Keine mit Timestamp erkennbaren Sessions gefunden. Wähle 'Alle Sessions' oder 'Letzte Tage'.")
        selected_csvs = []
    else:
        parsed_sorted = sorted(parsed, key=lambda x: x[1])
        max_n = len(parsed_sorted)
        default_n = min(30, max_n)
        n_sel = st.number_input("Anzahl neuester Sessions", min_value=1, max_value=max_n, value=default_n, step=1)
        selected_csvs = [p for p,_ in parsed_sorted[-int(n_sel):]]
        st.write(f"{len(selected_csvs)} Sessions ausgewählt (neueste {n_sel}).")

else:
    # Alle Sessions: show count and offer highlight config
    st.write(f"{len(selected_csvs)} Sessions (Alle)")

    with st.expander("Hervorhebung einstellen (letzte Tage / neueste N)", expanded=False):
        days_opt2 = st.selectbox("Markiere Sessions der letzten Tage (Keine = aus)",
                                ["Keine", "1 Tag", "3 Tage", "7 Tage", "14 Tage"], index=0)
        days_map2 = {"Keine": 0, "1 Tag":1, "3 Tage":3, "7 Tage":7, "14 Tage":14}
        highlight_days = days_map2[days_opt2]

        if n_csv > 0:
            highlight_newest_n = st.number_input("Markiere neueste N Sessions (0 = aus)",
                                                min_value=0, max_value=n_csv, value=0, step=1)
        else:
            highlight_newest_n = 0
            st.write("Keine CSVs gefunden, daher keine Markierung möglich.")

    # compute highlight sets
    recent_set = set()
    newest_set = set()
    if highlight_days and highlight_days > 0:
        now = datetime.now()
        for p in selected_csvs:
            dt = parse_dt_from_path(p)
            if dt is not None and dt >= (now - timedelta(days=highlight_days)):
                recent_set.add(os.path.basename(p))
    if highlight_newest_n and highlight_newest_n > 0:
        parsed = [(p, parse_dt_from_path(p)) for p in selected_csvs]
        parsed = [t for t in parsed if t[1] is not None]
        parsed_sorted = sorted(parsed, key=lambda x: x[1])
        newest = [os.path.basename(p) for p,_ in parsed_sorted[-int(highlight_newest_n):]]
        newest_set.update(newest)

    st.write("Sessions (Vorschau):")
    for p in sorted(selected_csvs):
        b = os.path.basename(p)
        badge_html = ""
        if b in recent_set and b in newest_set:
            badge_html = "<span class='badge badge-both'>beide</span>"
        elif b in recent_set:
            badge_html = "<span class='badge badge-recent'>letzte Tage</span>"
        elif b in newest_set:
            badge_html = "<span class='badge badge-newest'>neueste N</span>"
        st.markdown(f"<div class='session-row'>{badge_html}{b}</div>", unsafe_allow_html=True)

# optional limiter for large selections
limit_sessions = st.checkbox("Beschränke Verarbeitung zusätzlich auf neueste N Sessions (beschleunigt)", value=False)
if limit_sessions and len(selected_csvs)>0:
    default_n2 = min(100, len(selected_csvs))
    sel_n2 = st.number_input(f"Wenn aktiv: verarbeite nur die neuesten N aus der Auswahl (max {len(selected_csvs)})",
                             min_value=1, max_value=len(selected_csvs), value=default_n2, step=1)
    if sel_n2 < len(selected_csvs):
        parsed2 = [(p, parse_dt_from_path(p)) for p in selected_csvs]
        parsed2 = [t for t in parsed2 if t[1] is not None]
        parsed2_sorted = sorted(parsed2, key=lambda x: x[1])
        selected_csvs = [p for p,_ in parsed2_sorted[-int(sel_n2):]]
        st.write(f"Verarbeite jetzt {len(selected_csvs)} Sessions (neueste {sel_n2} aus Auswahl).")

# ---------------- Run analysis ----------------
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
        st.error("Keine CSVs zum Verarbeiten gefunden. Prüfe das Paket oder lade das ZIP neu hoch.")
    else:
        tmpdir = tempfile.mkdtemp(prefix="eeg_proc_")
        container = st.empty()
        df = build_session_table_from_list(resolved_selected, tmpdir, fs=fs if do_preproc else 0.0, st_container=container)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

        if df.empty:
            st.error("Keine gültigen Sessions (mit Bandspalten) gefunden. Prüfe Dateiformat.")
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

            df_out = df.copy()
            df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.download_button("Summary CSV herunterladen", data=df_out.to_csv(index=False).encode("utf-8"),
                               file_name="summary_indices.csv", mime="text/csv")
