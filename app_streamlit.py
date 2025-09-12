#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EEG-Auswertung – Datei-Upload + Analyse + PNG-Export (Plotly+Kaleido)

import os, re, glob, zipfile, tempfile, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Optional: Preprocessing
try:
    from scipy.signal import iirnotch, butter, filtfilt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Optional: PNG-Export Engine
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False


# ---------- Streamlit ----------
st.set_page_config(page_title="EEG-Auswertung", layout="wide")
st.title("EEG-Auswertung")
st.caption("Datei-Upload (ZIP/SIP/CSV) → Entpacken → Auswertung → Export als PNG")

# Session-Defaults
st.session_state.setdefault("workdir", tempfile.mkdtemp(prefix="eeg_works_"))
st.session_state.setdefault("df_summary", None)
st.session_state.setdefault("render_path", "")

workdir = st.session_state["workdir"]


# ---------- Helper ----------
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
    if not HAS_SCIPY: return x
    b,a = iirnotch(f0, Q, fs)
    return filtfilt(b,a,x)

def bandpass_signal(x, fs=250.0, low=0.5, high=45.0, order=4):
    if not HAS_SCIPY: return x
    nyq = fs/2.0
    b,a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b,a,x)

def preprocess_csv_if_raw(csv_path, out_tmp_dir, fs=250.0):
    # Nur filtern, wenn KEINE fertigen Bandspalten vorliegen
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return csv_path, False
    band_prefixes = ("Delta_","Theta_","Alpha_","Beta_","Gamma_")
    non_band = [c for c in df.columns if not str(c).startswith(band_prefixes)]
    numeric = [c for c in non_band if np.issubdtype(df[c].dtype, np.number)]
    if len(numeric) < 2 or len(df) < 10 or not HAS_SCIPY or fs <= 0:
        return csv_path, False
    proc = df.copy()
    for c in numeric:
        try:
            sig = proc[c].astype(float).values
            sig = notch_filter_signal(sig, fs=fs)
            sig = bandpass_signal(sig, fs=fs)
            proc[c] = sig
        except Exception:
            pass
    outp = os.path.join(out_tmp_dir, os.path.basename(csv_path).replace(".csv","_proc.csv"))
    proc.to_csv(outp, index=False)
    return outp, True

def load_session_relatives(csv_path):
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    bands = ["Delta","Theta","Alpha","Beta","Gamma"]
    cols = {b: [c for c in df.columns if str(c).startswith(f"{b}_")] or ([b] if b in df.columns else []) for b in bands}
    if not all(cols[b] for b in bands):
        return None
    sums = {}
    for b in bands:
        try:
            sums[b.lower()] = df[cols[b]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        except Exception:
            return None
    rel = pd.DataFrame(sums).replace([np.inf,-np.inf], np.nan).dropna()
    tot = rel.sum(axis=1).replace(0,np.nan)
    return rel.div(tot, axis=0).dropna()

def _is_good_rel(df):
    return isinstance(df, pd.DataFrame) and (not df.empty) and \
           all(c in df.columns for c in ["alpha","beta","theta","delta","gamma"])

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

def plot_stress_relax(df, smooth=5):
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
                  value_vars=["delta_trend","theta_trend","alpha_trend",
                              "beta_trend","gamma_trend","stresswave_trend","relaxwave_trend"],
                  var_name="Band", value_name="Wert")
    mapn = {"delta_trend":"Delta","theta_trend":"Theta","alpha_trend":"Alpha","beta_trend":"Beta",
            "gamma_trend":"Gamma","stresswave_trend":"Stress-Welle (Beta+Gamma)",
            "relaxwave_trend":"Entspannungs-Welle (Alpha+Theta)"}
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
            if arch.endswith(".extracted"):
                continue
            try:
                target = os.path.join(os.path.dirname(arch),
                                      os.path.splitext(os.path.basename(arch))[0] + "_extracted")
                os.makedirs(target, exist_ok=True)
                with zipfile.ZipFile(arch,"r") as zf:
                    zf.extractall(target)
                try:
                    os.rename(arch, arch+".extracted")
                except Exception:
                    try: os.remove(arch)
                    except Exception: pass
                changed = True
            except zipfile.BadZipFile:
                continue
            except Exception:
                continue

def make_beauty_figure(df, kind="stress_relax", smooth=5):
    x = df["date_str"]
    if kind == "stress_relax":
        d = df.copy()
        d["stress_trend"] = d["stress"].rolling(smooth, center=True, min_periods=1).mean()
        d["relax_trend"]  = d["relax"].rolling(smooth, center=True, min_periods=1).mean()
        d["stress_std"]   = d["stress"].rolling(smooth, center=True, min_periods=1).std().fillna(0)
        d["relax_std"]    = d["relax"].rolling(smooth, center=True, min_periods=1).std().fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"] + d["stress_std"],
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"] - d["stress_std"],
                                 fill='tonexty', fillcolor='rgba(220,70,70,0.18)',
                                 line=dict(width=0), name="Stress Band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"],
                                 line=dict(color='rgb(220,70,70)', width=4),
                                 name="Stress (Trend)", mode="lines+markers",
                                 marker=dict(size=6)))

        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"] + d["relax_std"],
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"] - d["relax_std"],
                                 fill='tonexty', fillcolor='rgba(70,170,70,0.18)',
                                 line=dict(width=0), name="Entspannung Band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"],
                                 line=dict(color='rgb(70,170,70)', width=4),
                                 name="Entspannung (Trend)", mode="lines+markers",
                                 marker=dict(size=6)))

        fig.update_layout(height=640, margin=dict(l=40,r=20,t=60,b=60),
                          title="Stress- und Entspannungs-Trend mit Schattierung",
                          xaxis=dict(type="category", tickangle=45, title="Datum"),
                          yaxis=dict(title="Index"))
        return fig

    elif kind == "bands":
        d = df.copy()
        for c in ["delta","theta","alpha","beta","gamma"]:
            d[f"{c}_trend"] = d[c].rolling(smooth, center=True, min_periods=1).mean()

        fig = go.Figure()
        palette = {
            "delta_trend":  "rgb(100,149,237)",
            "theta_trend":  "rgb(72,61,139)",
            "alpha_trend":  "rgb(34,139,34)",
            "beta_trend":   "rgb(255,165,0)",
            "gamma_trend":  "rgb(220,20,60)",
        }
        for key,color in palette.items():
            fig.add_trace(go.Scatter(
                x=x, y=d[key], name=key.replace("_trend","").capitalize(),
                line=dict(color=color, width=4), mode="lines"
            ))
        fig.update_layout(height=640, margin=dict(l=40,r=20,t=60,b=60),
                          title="EEG-Bänder (Trendlinien)",
                          xaxis=dict(type="category", tickangle=45, title="Datum"),
                          yaxis=dict(title="Relativer Anteil", range=[0,1]))
        return fig

    else:
        raise ValueError("Unknown kind")

def ensure_exports_dir():
    outdir = os.path.join(workdir, "exports")
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ---------- 1) Datei-Upload ----------
st.subheader("1) Datei-Upload")
uploads = st.file_uploader(
    "Dateien hochladen (ZIP/SIP mit CSVs oder einzelne CSVs)",
    type=["zip","sip","csv"],
    accept_multiple_files=True
)
if uploads:
    imported, extracted = 0, 0
    for up in uploads:
        fname = up.name
        local_path = os.path.join(workdir, fname)
        with open(local_path, "wb") as f:
            f.write(up.getbuffer())
        imported += 1
        if fname.lower().endswith((".zip",".sip")):
            try:
                with zipfile.ZipFile(local_path, "r") as zf:
                    zf.extractall(os.path.join(workdir, os.path.splitext(fname)[0] + "_extracted"))
                extracted += 1
            except zipfile.BadZipFile:
                st.warning(f"Beschädigtes Archiv übersprungen: {fname}")
    recursively_extract_archives(workdir)
    st.success(f"{imported} Datei(en) übernommen, {extracted} Archiv(e) entpackt.")


# ---------- 2) Parameter / QC ----------
st.subheader("2) Parameter / QC")
with st.expander("Hilfe zu Parametern", expanded=False):
    st.markdown("""
**Glättungsfenster**: Sessions für Trend-Glättung (3–7).  
**Sampling-Rate**: Nur für Rohdaten-Preprocessing nötig.  
**Export**: Unten „PNG rendern“ nutzen.
""")
smooth = st.slider("Glättungsfenster (Sessions)", 3, 11, 5, 2)
fs = st.number_input("Sampling-Rate für Preprocessing (Hz)", value=250.0, step=1.0)
do_preproc = st.checkbox("Preprocessing (Notch+Bandpass), falls Rohdaten", value=(True and HAS_SCIPY))

# Überblick über CSVs im Workdir
csv_paths_all = [p for p in glob.glob(os.path.join(workdir,"**","*.csv"), recursive=True)]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all)/(1024*1024) if n_csv>0 else 0.0
st.info(f"Gefundene CSVs: {n_csv} — Gesamtgröße: {total_mb:.1f} MB")


# ---------- 3) Auswertung ----------
if st.button("Auswertung starten"):
    recursively_extract_archives(workdir)
    selected_csvs = [p for p in glob.glob(os.path.join(workdir,"**","*.csv"), recursive=True)]
    st.info(f"Endgültige Anzahl zu verarbeitender Sessions: {len(selected_csvs)}")

    rows, failed = [], []
    total = max(1, len(selected_csvs))
    prog = st.progress(0)
    stat = st.empty()
    tmpdir = tempfile.mkdtemp(prefix="eeg_proc_")

    for i, cp in enumerate(sorted(selected_csvs), start=1):
        # CSV-Test
        try:
            _ = pd.read_csv(cp, nrows=1)
        except Exception:
            failed.append({"source": os.path.basename(cp), "reason": "Keine CSV (evtl. ZIP)."})
            stat.text(f"Skipping (no CSV): {os.path.basename(cp)}")
            continue

        dt = parse_dt_from_path(cp) or parse_dt_from_path(os.path.dirname(cp))
        if dt is None:
            failed.append({"source": os.path.basename(cp), "reason": "Ungültiger Zeitstempel im Namen."})
            stat.text(f"Skipping (no timestamp): {os.path.basename(cp)}")
            continue

        # Preproc bei Rohdaten
        proc_path = cp
        if do_preproc:
            proc_path, _ = preprocess_csv_if_raw(cp, tmpdir, fs=fs)

        rel = load_session_relatives(proc_path)
        if not _is_good_rel(rel):
            rel = load_session_relatives(cp)

        if not _is_good_rel(rel):
            failed.append({"source": os.path.basename(cp),
                           "reason": "Keine gültigen Bandspalten (Delta/Theta/Alpha/Beta/Gamma)."})
            stat.text(f"Skipping (no bands): {os.path.basename(cp)}")
            continue

        alpha, beta  = float(rel["alpha"].mean()), float(rel["beta"].mean())
        theta, delta = float(rel["theta"].mean()), float(rel["delta"].mean())
        gamma        = float(rel["gamma"].mean())
        rows.append({
            "datetime": dt, "alpha":alpha,"beta":beta,"theta":theta,"delta":delta,"gamma":gamma,
            "stress": beta/(alpha+1e-9), "relax": alpha/(beta+1e-9), "source": os.path.basename(cp)
        })
        prog.progress(int(i/total*100)); stat.text(f"Processed {i}/{total}: {os.path.basename(cp)}")

    prog.empty(); stat.empty()
    try: shutil.rmtree(tmpdir)
    except Exception: pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("datetime").reset_index(drop=True)
        df["date_str"] = df["datetime"].dt.strftime("%d-%m-%y %H:%M")

    # Ergebnis in Session parken
    st.session_state["df_summary"] = df.copy() if (isinstance(df, pd.DataFrame) and not df.empty) else None

    if st.session_state["df_summary"] is not None:
        dfv = st.session_state["df_summary"]
        if len(dfv) == 1:
            st.subheader("Einzel-Session")
            st.plotly_chart(plot_single_session_interactive(dfv), use_container_width=True)
            st.dataframe(dfv.round(4))
        else:
            st.subheader("Stress/Entspannung")
            st.plotly_chart(plot_stress_relax(dfv, smooth=smooth), use_container_width=True)
            st.subheader("Bänder + Wellen")
            st.plotly_chart(plot_bands(dfv, smooth=smooth), use_container_width=True)
            st.subheader("Tabelle")
            st.dataframe(dfv.round(4))

        df_out = dfv.copy()
        df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.download_button("Summary CSV herunterladen",
                           data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="summary_indices.csv", mime="text/csv")
    else:
        st.warning("Keine verwertbaren Sessions.")

    if failed:
        st.subheader("Folgende Dateien wurden übersprungen:")
        for f in failed:
            st.warning(f"{f['source']}: {f['reason']}")


# ---------- 4) Export (PNG) – unabhängig von „Auswertung starten“ ----------
st.subheader("Export")
st.caption(f"Kaleido installiert: {HAS_KALEIDO}")
df_render = st.session_state.get("df_summary", None)

if df_render is None or df_render.empty:
    st.info("Keine Auswertung im Speicher. Klicke zuerst auf „Auswertung starten“.")
else:
    st.caption(f"Auswertung im Speicher: {len(df_render)} Zeile(n)")
    render_kind = st.selectbox("Motiv", ["Stress/Entspannung (Trend)", "Bänder (Trend)"], key="render_kind")
    clicked = st.button("PNG rendern", key="render_btn")

    if clicked:
        try:
            if not HAS_KALEIDO:
                raise RuntimeError("Kaleido nicht installiert. Füge `kaleido>=0.2` in requirements.txt hinzu.")
            kind = "stress_relax" if "Stress" in render_kind else "bands"
            fig  = make_beauty_figure(df_render, kind=kind, smooth=smooth)
            outdir = ensure_exports_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            png_path = os.path.join(outdir, f"render_{kind}_{ts}.png")
            pio.write_image(fig, png_path, width=1600, height=900, scale=3, engine="kaleido")
            st.session_state["render_path"] = png_path
            st.success(f"PNG erzeugt: {png_path}")
        except Exception as e:
            st.error("PNG-Rendering fehlgeschlagen.")
            st.exception(e)

    p = st.session_state.get("render_path", "")
    if p and os.path.isfile(p):
        st.image(p, caption="Rendering-Vorschau (PNG)", use_column_width=True)
        with open(p, "rb") as f:
            st.download_button("PNG herunterladen", f, file_name=os.path.basename(p), mime="image/png")


# ---------- Wartung ----------
with st.expander("Debug / Wartung", expanded=False):
    if st.button("Arbeitsordner leeren"):
        try:
            shutil.rmtree(st.session_state["workdir"])
        except Exception:
            pass
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="eeg_works_")
        st.session_state["df_summary"] = None
        st.session_state["render_path"] = ""
        st.success("Arbeitsordner geleert. Seite neu laden oder neue Dateien hochladen.")
