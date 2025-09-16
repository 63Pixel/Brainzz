#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EEG-Auswertung – Upload, Auswertung, persistente Charts, PNG-Rendering (Plotly→Kaleido, Fallback Matplotlib)

import os, re, glob, zipfile, tempfile, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# optional: Preprocessing
try:
    from scipy.signal import iirnotch, butter, filtfilt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# optional: Plotly→Kaleido PNG
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# Matplotlib-Fallback (kein Chrome nötig)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Streamlit ----------
st.set_page_config(page_title="EEG-Auswertung", layout="wide")
st.title("EEG-Auswertung")
st.caption("Datei-Upload (ZIP/SIP/CSV) → Entpacken → Auswertung → Export als PNG")


# ---------- Persistentes Arbeitsverzeichnis ----------
def get_workdir():
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="eeg_works_")
    return st.session_state["workdir"]

workdir = get_workdir()


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
    """Filter nur, wenn KEINE fertigen Bandspalten vorliegen."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return csv_path, False
    band_prefixes = ("Delta_", "Theta_", "Alpha_", "Beta_", "Gamma_")
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
    outp = os.path.join(out_tmp_dir, os.path.basename(csv_path).replace(".csv", "_proc.csv"))
    proc.to_csv(outp, index=False)
    return outp, True

def load_session_relatives(csv_path, agg="power"):
    """Lädt relative Bandanteile (Delta..Gamma) zeitaufgelöst aus einer CSV mit Kanälen Delta_*, Theta_* usw.
       Gibt einen DataFrame mit Spalten ['delta','theta','alpha','beta','gamma'] (0..1, zeitsynchron) zurück.
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None

    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    cols = {b: [c for c in df.columns if str(c).startswith(f"{b}_")] or ([b] if b in df.columns else []) for b in bands}
    if not all(cols[b] for b in bands):
        return None

    out = {}
    for b in bands:
        try:
            val = df[cols[b]].apply(pd.to_numeric, errors="coerce")
            if agg == "abs":
                val = val.abs()
            elif (val < 0).any().any():
                val = val.pow(2)  # power
            out[b.lower()] = val.mean(axis=1)  # Kanäle mitteln
        except Exception:
            return None

    rel = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
    tot = rel.sum(axis=1).replace(0, np.nan)
    return rel.div(tot, axis=0).dropna()

def _is_good_rel(df):
    return isinstance(df, pd.DataFrame) and not df.empty and \
           all(c in df.columns for c in ["alpha", "beta", "theta", "delta", "gamma"])

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
                target = os.path.join(os.path.dirname(arch),
                                      os.path.splitext(os.path.basename(arch))[0] + "_extracted")
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

# Rolling-Helfer: 1 → keine Glättung
def roll_mean(s, w):  # w >= 1
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).mean()

def roll_std(s, w):
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).std()


# ---------- Interaktive Plotly-Anzeigen (Sessions-Übersicht) ----------
def plot_single_session_interactive(df):
    vals = {"Stress": df["stress"].iloc[0], "Entspannung": df["relax"].iloc[0],
            "Delta": df["delta"].iloc[0], "Theta": df["theta"].iloc[0],
            "Alpha": df["alpha"].iloc[0], "Beta": df["beta"].iloc[0],
            "Gamma": df["gamma"].iloc[0]}
    data = pd.DataFrame({"Metrik": list(vals.keys()), "Wert": list(vals.values())})
    fig = px.bar(data, x="Metrik", y="Wert", color="Metrik", text="Wert", height=320)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", showlegend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig

def plot_stress_relax(df, smooth=1):
    d = df.copy()
    d["stress_trend"] = roll_mean(d["stress"], smooth)
    d["relax_trend"]  = roll_mean(d["relax"],  smooth)
    long = d.melt(id_vars=["date_str"], value_vars=["stress", "relax", "stress_trend", "relax_trend"],
                  var_name="Metrik", value_name="Wert")
    fig = px.line(long, x="date_str", y="Wert", color="Metrik", markers=True, height=360)
    fig.update_layout(xaxis=dict(type="category"))
    return fig

def plot_bands(df, smooth=1, y_mode="0–1 (fix)"):
    d = df.copy()
    d["stresswave"] = d["beta"] + d["gamma"]
    d["relaxwave"]  = d["alpha"] + d["theta"]
    for c in ["delta", "theta", "alpha", "beta", "gamma", "stresswave", "relaxwave"]:
        d[f"{c}_trend"] = roll_mean(d[c], smooth)

    long = d.melt(
        id_vars=["date_str"],
        value_vars=["delta_trend", "theta_trend", "alpha_trend", "beta_trend", "gamma_trend",
                    "stresswave_trend", "relaxwave_trend"],
        var_name="Band", value_name="Wert"
    )
    name_map = {
        "delta_trend": "Delta", "theta_trend": "Theta", "alpha_trend": "Alpha", "beta_trend": "Beta",
        "gamma_trend": "Gamma", "stresswave_trend": "Stress-Welle (Beta+Gamma)",
        "relaxwave_trend": "Entspannungs-Welle (Alpha+Theta)"
    }
    long["Band"] = long["Band"].map(name_map)

    if y_mode == "Abweichung vom Mittelwert (%)":
        long["Wert"] = long.groupby("Band")["Wert"].transform(lambda s: (s / (s.mean() + 1e-12) - 1.0) * 100.0)

    fig = px.line(long, x="date_str", y="Wert", color="Band", markers=True, height=380)
    fig.update_layout(xaxis=dict(type="category"))

    if y_mode == "0–1 (fix)":
        fig.update_yaxes(range=[0, 1], title="Wert")
    elif y_mode == "Auto (zoom)":
        fig.update_yaxes(autorange=True, title="Wert")
    else:
        fig.update_yaxes(title="Δ zum Mittelwert [%]")
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")

    return fig


# ---------- Zusatz: Einzel-Session mit Zeitachse ----------
def find_csv_by_basename(name: str, root_dir: str):
    """Suche die Original-CSV im Arbeitsordner (rekursiv) anhand des Dateinamens."""
    paths = [p for p in glob.glob(os.path.join(root_dir, "**", name), recursive=True)]
    return paths[0] if paths else None

def plot_single_session_timeline(csv_path, fs=250.0, smooth_seconds=3, y_mode="0–1 (fix)"):
    """Zeitlicher Verlauf der relativen Bänder (Delta..Gamma) + Stress/Relax-Wellen über die Session."""
    rel = load_session_relatives(csv_path)
    if rel is None or rel.empty:
        return go.Figure()

    # Versuche Zeitachse aus CSV zu lesen; sonst aus fs approximieren
    try:
        df0 = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df0 = None

    t = None
    if df0 is not None:
        cand = [c for c in df0.columns if str(c).lower() in
                ["timestamp", "time", "timesec", "t", "elapsed", "seconds", "secs",
                 "ms", "millis", "datetime", "date_time", "date", "zeit"]]
        for c in cand:
            s = df0[c]
            if np.issubdtype(s.dtype, np.number):
                val = pd.to_numeric(s, errors="coerce").values.astype(float)
                t = val/1000.0 if ("ms" in c.lower() or "millis" in c.lower()) else val
                break
            else:
                try:
                    td = pd.to_datetime(s, errors="coerce")
                    if td.notna().any():
                        t = (td - td.iloc[0]).dt.total_seconds().values
                        break
                except Exception:
                    pass
    if t is None:
        t = np.arange(len(rel)) / fs if fs and fs > 0 else np.arange(len(rel))

    # Längen angleichen (DropNAs in rel können kürzen)
    n = min(len(rel), len(t))
    rel = rel.iloc[:n].copy()
    t = t[:n]

    # Glättung in Sekunden → Samples
    w = max(1, int(round((smooth_seconds if smooth_seconds else 0) * (fs if fs else 1))))
    for c in ["delta", "theta", "alpha", "beta", "gamma"]:
        if c in rel.columns:
            rel[c] = roll_mean(rel[c], w)
    rel["stresswave"] = roll_mean(rel["beta"] + rel["gamma"], w)
    rel["relaxwave"]  = roll_mean(rel["alpha"] + rel["theta"], w)

    long = rel.copy()
    long["t"] = t
    long = long.melt(id_vars=["t"],
                     value_vars=["delta","theta","alpha","beta","gamma","stresswave","relaxwave"],
                     var_name="Band", value_name="Wert")
    name_map = {
        "delta": "Delta", "theta": "Theta", "alpha": "Alpha", "beta": "Beta", "gamma": "Gamma",
        "stresswave": "Stress-Welle (Beta+Gamma)",
        "relaxwave": "Entspannungs-Welle (Alpha+Theta)"
    }
    long["Band"] = long["Band"].map(name_map)

    fig = px.line(long, x="t", y="Wert", color="Band", height=380)
    fig.update_layout(xaxis_title="Zeit [s]")
    if y_mode == "0–1 (fix)":
        fig.update_yaxes(range=[0, 1], title="Relativer Anteil")
    else:
        fig.update_yaxes(title="Relativer Anteil")
    return fig


# ---------- „Schönes Rendering“ (Plotly mit Punkt-Schatten) ----------
def make_beauty_figure(df, kind="stress_relax", smooth=1):
    x = df["date_str"]
    fig = go.Figure()

    if kind == "stress_relax":
        d = df.copy()
        d["stress_trend"] = roll_mean(d["stress"], smooth)
        d["relax_trend"]  = roll_mean(d["relax"],  smooth)
        d["stress_std"]   = roll_std(d["stress"], smooth).fillna(0)
        d["relax_std"]    = roll_std(d["relax"],  smooth).fillna(0)

        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"] + d["stress_std"], line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"] - d["stress_std"], fill='tonexty',
                                 fillcolor='rgba(220,70,70,0.18)', line=dict(width=0),
                                 name="Stress Band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=14, color="rgba(0,0,0,0.20)")))
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"], name="Stress (Trend)", mode="lines+markers",
                                 line=dict(color='rgb(220,70,70)', width=4),
                                 marker=dict(size=7, color='rgb(220,70,70)')))

        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"] + d["relax_std"], line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"] - d["relax_std"], fill='tonexty',
                                 fillcolor='rgba(70,170,70,0.18)', line=dict(width=0),
                                 name="Entspannung Band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=14, color="rgba(0,0,0,0.20)")))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"], name="Entspannung (Trend)", mode="lines+markers",
                                 line=dict(color='rgb(70,170,70)', width=4),
                                 marker=dict(size=7, color='rgb(70,170,70)')))

        fig.update_layout(height=640, margin=dict(l=40, r=20, t=60, b=60),
                          title="Stress- und Entspannungs-Trend",
                          xaxis=dict(type="category", tickangle=45, title="Datum"),
                          yaxis=dict(title="Index"))
        return fig

    if kind == "bands":
        d = df.copy()
        for c in ["delta", "theta", "alpha", "beta", "gamma"]:
            d[f"{c}_trend"] = roll_mean(d[c], smooth)
        palette = {
            "delta_trend": ("Delta", "rgb(100,149,237)"),
            "theta_trend": ("Theta", "rgb(72,61,139)"),
            "alpha_trend": ("Alpha", "rgb(34,139,34)"),
            "beta_trend":  ("Beta",  "rgb(255,165,0)"),
            "gamma_trend": ("Gamma", "rgb(220,20,60)"),
        }
        for key, (label, color) in palette.items():
            fig.add_trace(go.Scatter(x=x, y=d[key], mode="markers", hoverinfo="skip", showlegend=False,
                                     marker=dict(size=12, color="rgba(0,0,0,0.18)")))
            fig.add_trace(go.Scatter(x=x, y=d[key], name=label, mode="lines+markers",
                                     line=dict(color=color, width=3.5),
                                     marker=dict(size=6, color=color)))

        fig.update_layout(height=640, margin=dict(l=40, r=20, t=60, b=60),
                          title="EEG-Bänder (Trendlinien)",
                          xaxis=dict(type="category", tickangle=45, title="Datum"),
                          yaxis=dict(title="Relativer Anteil", range=[0, 1]))
        return fig

    raise ValueError("Unknown kind")


# ---------- Matplotlib-Fallback-Rendering ----------
def render_png_matplotlib(df, kind="stress_relax", smooth=1, outpath="render.png"):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    x = np.arange(len(df))
    xticks = df["date_str"].tolist()

    if kind == "stress_relax":
        d = df.copy()
        d["stress_trend"] = roll_mean(d["stress"], smooth)
        d["relax_trend"]  = roll_mean(d["relax"],  smooth)
        d["stress_std"]   = roll_std(d["stress"], smooth).fillna(0)
        d["relax_std"]    = roll_std(d["relax"],  smooth).fillna(0)

        ax.fill_between(x, d["stress_trend"]-d["stress_std"], d["stress_trend"]+d["stress_std"],
                        alpha=0.20, color=(0.86, 0.27, 0.27))
        ax.fill_between(x, d["relax_trend"]-d["relax_std"], d["relax_trend"]+d["relax_std"],
                        alpha=0.20, color=(0.27, 0.67, 0.27))
        ax.scatter(x, d["stress_trend"], s=180, c="k", alpha=0.20, zorder=2)
        ax.scatter(x, d["relax_trend"],  s=180, c="k", alpha=0.20, zorder=2)
        ax.plot(x, d["stress_trend"], c=(0.86, 0.27, 0.27), lw=3.5, zorder=3, label="Stress (Trend)")
        ax.plot(x, d["relax_trend"],  c=(0.27, 0.67, 0.27), lw=3.5, zorder=3, label="Entspannung (Trend)")
        ax.legend(loc="best"); ax.set_ylabel("Index"); ax.set_title("Stress- und Entspannungs-Trend")

    elif kind == "bands":
        d = df.copy()
        for c in ["delta", "theta", "alpha", "beta", "gamma"]:
            d[f"{c}_trend"] = roll_mean(d[c], smooth)
        series = [
            ("Delta", d["delta_trend"], (0.39, 0.58, 0.93)),
            ("Theta", d["theta_trend"], (0.28, 0.24, 0.55)),
            ("Alpha", d["alpha_trend"], (0.13, 0.55, 0.13)),
            ("Beta",  d["beta_trend"],  (1.00, 0.65, 0.00)),
            ("Gamma", d["gamma_trend"], (0.86, 0.08, 0.24)),
        ]
        for label, y, col in series:
            ax.scatter(x, y, s=160, c="k", alpha=0.18, zorder=2)
            ax.plot(x, y, lw=3, c=col, zorder=3, label=label)
            ax.scatter(x, y, s=48, c=col, zorder=4)
        ax.set_ylim(0, 1); ax.set_ylabel("Relativer Anteil")
        ax.set_title("EEG-Bänder (Trendlinien)"); ax.legend(loc="best")

    ax.set_xticks(x); ax.set_xticklabels(xticks, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    return outpath


# ---------- Chart-Baukasten + Cache ----------
def build_charts(df: pd.DataFrame, smooth: int, y_mode: str):
    charts = {}
    if len(df) == 1:
        charts["single"] = plot_single_session_interactive(df)
    else:
        charts["stress"] = plot_stress_relax(df, smooth=smooth)
        charts["bands"]  = plot_bands(df, smooth=smooth, y_mode=y_mode)
    return charts


# ---------- 1) Datei-Upload ----------
st.subheader("1) Datei-Upload")
uploads = st.file_uploader(
    "Dateien hochladen (ZIP/SIP mit CSVs oder einzelne CSVs)",
    type=["zip", "sip", "csv"],
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
        if fname.lower().endswith((".zip", ".sip")):
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
**Glättungsfenster**: 1 = keine Glättung; höhere Werte glätten stärker.  
**Sampling-Rate**: Nur für Rohdaten-Preprocessing nötig.
""")
smooth = st.slider("Glättungsfenster (Sessions)", 1, 15, 2, 1,
                   help="1 = keine Glättung; höhere Werte glätten stärker")
y_mode = st.selectbox("Y-Achse für Bänder", ["0–1 (fix)", "Auto (zoom)", "Abweichung vom Mittelwert (%)"], index=0)
fs = st.number_input("Sampling-Rate für Preprocessing (Hz) / Timeline (Fallback)", value=250.0, step=1.0)
do_preproc = st.checkbox("Preprocessing (Notch+Bandpass), falls Rohdaten", value=(True and HAS_SCIPY))

# CSV-Überblick
csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*.csv"), recursive=True)]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all) / (1024*1024) if n_csv > 0 else 0.0
st.info(f"Gefundene CSVs: {n_csv} — Gesamtgröße: {total_mb:.1f} MB")

# Charts bei Parameterwechsel neu bauen
if "df_summary" in st.session_state and not st.session_state["df_summary"].empty:
    if st.session_state.get("last_smooth") != smooth or st.session_state.get("last_y_mode") != y_mode:
        st.session_state["charts"] = build_charts(st.session_state["df_summary"], smooth, y_mode)
        st.session_state["last_smooth"] = smooth
        st.session_state["last_y_mode"]  = y_mode


# ---------- 3) Auswertung ----------
if st.button("Auswertung starten"):
    recursively_extract_archives(workdir)
    selected_csvs = [p for p in glob.glob(os.path.join(workdir, "**", "*.csv"), recursive=True)]
    if not selected_csvs:
        st.error("Keine CSVs gefunden.")
    else:
        tmpdir = tempfile.mkdtemp(prefix="eeg_proc_")
        rows, failed = [], []
        for cp in sorted(selected_csvs):
            try:
                _ = pd.read_csv(cp, nrows=1)
            except Exception:
                failed.append({"source": os.path.basename(cp), "reason": "Keine CSV (evtl. ZIP)."})
                continue
            dt = parse_dt_from_path(cp) or parse_dt_from_path(os.path.dirname(cp))
            if dt is None:
                failed.append({"source": os.path.basename(cp), "reason": "Ungültiger Zeitstempel."})
                continue
            proc_path, _ = preprocess_csv_if_raw(cp, tmpdir, fs=(fs if do_preproc else 0.0))
            rel = load_session_relatives(proc_path)
            if not _is_good_rel(rel):
                rel = load_session_relatives(cp)
            if not _is_good_rel(rel):
                failed.append({"source": os.path.basename(cp), "reason": "Keine gültigen Bandspalten."})
                continue
            alpha, beta  = float(rel["alpha"].mean()), float(rel["beta"].mean())
            theta, delta = float(rel["theta"].mean()), float(rel["delta"].mean())
            gamma        = float(rel["gamma"].mean())
            rows.append({
                "datetime": dt, "alpha": alpha, "beta": beta, "theta": theta, "delta": delta, "gamma": gamma,
                "stress": beta/(alpha+1e-9), "relax": alpha/(beta+1e-9), "source": os.path.basename(cp)
            })
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

        df = pd.DataFrame(rows)
        if df.empty:
            st.error("Keine gültigen Sessions.")
        else:
            df = df.sort_values("datetime").reset_index(drop=True)
            df["date_str"] = df["datetime"].dt.strftime("%d-%m-%y %H:%M")
            st.session_state["df_summary"] = df.copy()
            st.session_state["last_smooth"] = smooth
            st.session_state["last_y_mode"]  = y_mode
            st.session_state["charts"] = build_charts(df, smooth, y_mode)
            st.success(f"{len(df)} Session(s) ausgewertet. Anzeige unten aktualisiert.")
        if failed:
            st.subheader("Übersprungene Dateien")
            for f in failed:
                st.warning(f"{f['source']}: {f['reason']}")


# ---------- 3b) Anzeige aus Session-State ----------
df_show = st.session_state.get("df_summary", pd.DataFrame())
charts  = st.session_state.get("charts", {})

if not df_show.empty:
    if len(df_show) == 1:
        st.subheader("Einzel-Session")
        fig = charts.get("single") or plot_single_session_interactive(df_show)
        st.plotly_chart(fig, use_container_width=True)

        # Zeitverlauf (Einzel-Session) mit Zeitachse
        st.markdown("### Zeitverlauf (Einzel-Session)")
        ss_smooth = st.slider("Glättung (Sekunden)", 0, 30, 3, 1,
                              help="Glättet die zeitliche Kurve innerhalb der Session.")
        y_mode_single = st.selectbox("Y-Achse (Einzel-Session)", ["0–1 (fix)", "Auto (zoom)"], index=0)

        csv_name = df_show.iloc[0]["source"]
        csv_path = find_csv_by_basename(csv_name, workdir)
        if csv_path:
            fig_ts = plot_single_session_timeline(csv_path, fs=fs, smooth_seconds=ss_smooth, y_mode=y_mode_single)
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Original-CSV für die Einzel-Session wurde nicht gefunden.")

        st.dataframe(df_show.round(4))
    else:
        st.subheader("Stress/Entspannung")
        fig1 = charts.get("stress") or plot_stress_relax(df_show, smooth=smooth)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Bänder + Wellen")
        fig2 = charts.get("bands") or plot_bands(df_show, smooth=smooth, y_mode=y_mode)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Tabelle")
        st.dataframe(df_show.round(4))

    # Download Summary
    df_out = df_show.copy()
    df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button("Summary CSV herunterladen",
                       data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name="summary_indices.csv", mime="text/csv")


# ---------- 4) Export (PNG) ----------
st.subheader("Export")
if df_show.empty:
    st.info("Keine Auswertung im Speicher. Erst „Auswertung starten“.")
else:
    render_kind = st.selectbox("Motiv", ["Stress/Entspannung (Trend)", "Bänder (Trend)"], key="render_kind")
    render_btn  = st.button("PNG rendern", key="render_btn")
    st.session_state.setdefault("render_path", "")

    if render_btn:
        kind = "stress_relax" if "Stress" in render_kind else "bands"
        outdir = os.path.join(workdir, "exports"); os.makedirs(outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = os.path.join(outdir, f"render_{kind}_{ts}.png")
        try:
            if not HAS_KALEIDO:
                raise RuntimeError("Kaleido nicht installiert")
            fig = make_beauty_figure(df_show, kind=kind, smooth=smooth)
            pio.write_image(fig, png_path, width=1600, height=900, scale=3, engine="kaleido")
        except Exception:
            png_path = render_png_matplotlib(df_show, kind=kind, smooth=smooth, outpath=png_path)
        st.session_state["render_path"] = png_path
        st.success(f"PNG erzeugt: {png_path}")

    if st.session_state.get("render_path") and os.path.isfile(st.session_state["render_path"]):
        p = st.session_state["render_path"]
        st.image(p, caption="Rendering-Vorschau (PNG)", use_container_width=True)
        with open(p, "rb") as f:
            st.download_button("PNG herunterladen", f, file_name=os.path.basename(p), mime="image/png")


# ---------- Wartung ----------
with st.expander("Debug / Wartung", expanded=False):
    if st.button("Arbeitsordner leeren"):
        try:
            shutil.rmtree(st.session_state["workdir"])
        except Exception:
            pass
        for k in ["workdir", "df_summary", "charts", "render_path", "last_smooth", "last_y_mode"]:
            st.session_state.pop(k, None)
        st.success("Arbeitsordner geleert. Seite neu laden.")
