#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EEG-Auswertung – Upload, Auswertung, persistente Charts, JPG-Rendering (Plotly→Kaleido, Fallback Matplotlib)
#
# Änderungen:
# - Export als JPG (Qualität 80%)
# - Export-UI: wähle Timeline / Balken / Kombi (Timeline oben, Balken unten)
# - Timeline zeigt Timecode (Uhrzeit) am unteren Rand
# - Keine massenhaften Marker mehr (Linien + CI/Shadows)
# - Robust: Kaleido ohne engine= / Matplotlib-Fallback

import os
import re
import glob
import zipfile
import tempfile
import shutil
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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

# Pillow für PNG->JPG
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# Matplotlib-Fallback (kein Chrome nötig)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="EEG-Auswertung", layout="wide")
st.title("EEG-Auswertung")
st.caption("Datei-Upload (ZIP/SIP/CSV) → Entpacken → Auswertung → Export als JPG (Qualität 80%)")

# ---------------- Arbeitsverzeichnis ----------------
def get_workdir():
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="eeg_works_")
    return st.session_state["workdir"]

workdir = get_workdir()

# ---------------- Helfer ----------------
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
    """Lädt relative Bandanteile (Delta..Gamma) zeitaufgelöst."""
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
                val = val.pow(2)
            out[b.lower()] = val.mean(axis=1)
        except Exception:
            return None

    rel = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
    tot = rel.sum(axis=1).replace(0, np.nan)
    return rel.div(tot, axis=0).dropna()

def _is_good_rel(df):
    return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in ["alpha", "beta", "theta", "delta", "gamma"])

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

def roll_mean(s, w):
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).mean()

def roll_std(s, w):
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).std()

def find_csv_by_basename(name: str, root_dir: str):
    paths = [p for p in glob.glob(os.path.join(root_dir, "**", name), recursive=True)]
    return paths[0] if paths else None

# ---------------- Plot-Funktionen (leicht, performant: keine massiven Marker) ----------------
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
    long = d.melt(id_vars=["date_str"], value_vars=["stress_trend", "relax_trend"],
                  var_name="Metrik", value_name="Wert")
    fig = px.line(long, x="date_str", y="Wert", color="Metrik", markers=False, height=360)
    fig.update_layout(xaxis=dict(type="category"))
    return fig

def plot_bands(df, smooth=1, y_mode="0–1 (fix)"):
    d = df.copy()
    d["stresswave"] = d["beta"] + d["gamma"]
    d["relaxwave"]  = d["alpha"] + d["theta"]
    for c in ["delta", "theta", "alpha", "beta", "gamma", "stresswave", "relaxwave"]:
        d[f"{c}_trend"] = roll_mean(d[c], smooth)
        d[f"{c}_std"] = roll_std(d[c], smooth).fillna(0)

    palette = {
        "delta_trend": "rgb(100,149,237)",
        "theta_trend": "rgb(128,0,128)",
        "alpha_trend": "rgb(34,139,34)",
        "beta_trend":  "rgb(255,165,0)",
        "gamma_trend": "rgb(220,20,60)",
        "stresswave_trend": "rgb(220,120,60)",
        "relaxwave_trend": "rgb(60,200,180)"
    }

    x = d["date_str"]
    fig = go.Figure()
    name_map = {
        "delta_trend": "Delta", "theta_trend": "Theta", "alpha_trend": "Alpha", "beta_trend": "Beta",
        "gamma_trend": "Gamma", "stresswave_trend": "Stress-Welle (Beta+Gamma)",
        "relaxwave_trend": "Entspannungs-Welle (Alpha+Theta)"
    }

    for key, label in name_map.items():
        if key not in d:
            continue
        base = key.replace("_trend", "")
        ci_up = (d[key] + d[f"{base}_std"]).tolist()
        ci_dn = (d[key] - d[f"{base}_std"]).tolist()[::-1]
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=ci_up + ci_dn,
            fill="toself",
            fillcolor="rgba(150,150,150,0.08)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x, y=d[key], mode="lines",
            line=dict(color="rgba(0,0,0,0.06)", width=6),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x, y=d[key], mode="lines", name=label,
            line=dict(color=palette.get(key, "rgb(100,100,100)"), width=3.0)
        ))

    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=40), xaxis=dict(type="category"), xaxis_title="Datum")
    if y_mode == "0–1 (fix)":
        fig.update_yaxes(range=[0, 1], title="Wert")
    elif y_mode == "Auto (zoom)":
        fig.update_yaxes(autorange=True, title="Wert")
    else:
        fig.update_yaxes(title="Δ zum Mittelwert [%]")
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    return fig

# ---------------- Einzel-Session Timeline (Zeitcode unten) ----------------
def plot_single_session_timeline(csv_path, fs=250.0, smooth_seconds=3, y_mode="0–1 (fix)"):
    rel = load_session_relatives(csv_path)
    if rel is None or rel.empty:
        return go.Figure()

    # Try to load CSV and find time-like column
    try:
        df0 = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df0 = None

    x_vals = None
    is_datetime = False
    if df0 is not None:
        cand = [c for c in df0.columns if str(c).lower() in
                ["timestamp", "time", "timesec", "t", "elapsed", "seconds", "secs",
                 "ms", "millis", "datetime", "date_time", "date", "zeit", "uhrzeit", "clock"]]
        for c in cand:
            s = df0[c]
            if np.issubdtype(s.dtype, np.number):
                val = pd.to_numeric(s, errors="coerce").astype(float)
                # treat ms vs s
                if np.nanmax(val) > 1e6:
                    try:
                        x_vals = pd.to_datetime(val, unit="ms")
                        is_datetime = True
                        break
                    except Exception:
                        pass
                else:
                    try:
                        x_vals = pd.to_datetime(val, unit="s")
                        is_datetime = True
                        break
                    except Exception:
                        pass
            else:
                try:
                    td = pd.to_datetime(s, errors="coerce")
                    if td.notna().any():
                        x_vals = td
                        is_datetime = True
                        break
                except Exception:
                    pass

    # fallback: use seconds -> convert to datetime origin so Plotly shows timecodes nicely
    if x_vals is None:
        secs = np.arange(len(rel)) / (fs if fs and fs > 0 else 1.0)
        x_vals = pd.to_datetime(secs, unit="s", origin=pd.Timestamp("1970-01-01"))
        is_datetime = True

    # align lengths
    n = min(len(rel), len(x_vals))
    rel = rel.iloc[:n].copy()
    x_vals = pd.Series(x_vals).iloc[:n].reset_index(drop=True)

    # smoothing: seconds -> samples
    w = max(1, int(round((smooth_seconds if smooth_seconds else 0) * (fs if fs else 1))))
    for c in ["delta", "theta", "alpha", "beta", "gamma"]:
        if c in rel.columns:
            rel[c] = roll_mean(rel[c], w)
    rel["stresswave"] = roll_mean(rel["beta"] + rel["gamma"], w)
    rel["relaxwave"]  = roll_mean(rel["alpha"] + rel["theta"], w)

    # build traces (shadow + line) - no dense markers
    bands_map = {
        "delta": "Delta", "theta": "Theta", "alpha": "Alpha",
        "beta": "Beta", "gamma": "Gamma",
        "stresswave": "Stress-Welle (Beta+Gamma)", "relaxwave": "Entspannungs-Welle (Alpha+Theta)"
    }

    fig = go.Figure()
    palette = {
        "Delta": "rgb(100,149,237)",
        "Theta": "rgb(128,0,128)",
        "Alpha": "rgb(34,139,34)",
        "Beta": "rgb(255,165,0)",
        "Gamma": "rgb(220,20,60)",
        "Stress-Welle (Beta+Gamma)": "rgb(220,120,60)",
        "Entspannungs-Welle (Alpha+Theta)": "rgb(60,200,180)"
    }

    for key, label in bands_map.items():
        if key not in rel:
            continue
        y = rel[key].values
        # subtle shadow
        fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="rgba(0,0,0,0.06)", width=8)))
        # main line
        fig.add_trace(go.Scatter(x=x_vals, y=y, mode="lines", name=label,
                                 line=dict(color=palette.get(label, "gray"), width=2)))

    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=40))
    # Timecode at bottom (plotly default is bottom). Format if datetime-like
    if is_datetime:
        fig.update_xaxes(title_text="Uhrzeit", tickformat="%H:%M:%S", tickangle=45)
    else:
        fig.update_xaxes(title_text="Zeit [s]", tickangle=45)
    if y_mode == "0–1 (fix)":
        fig.update_yaxes(range=[0,1], title="Relativer Anteil")
    else:
        fig.update_yaxes(title="Relativer Anteil")
    return fig

# ---------------- make_beauty_figure (multi-session summary) ----------------
def make_beauty_figure(df, kind="stress_relax", smooth=1):
    x = df["date_str"]
    fig = go.Figure()

    if kind == "stress_relax":
        d = df.copy()
        d["stress_trend"] = roll_mean(d["stress"], smooth)
        d["relax_trend"]  = roll_mean(d["relax"],  smooth)
        d["stress_std"]   = roll_std(d["stress"], smooth).fillna(0)
        d["relax_std"]    = roll_std(d["relax"],  smooth).fillna(0)

        # Stress CI
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list((d["stress_trend"] + d["stress_std"])) + list((d["stress_trend"] - d["stress_std"])[::-1]),
            fill="toself", fillcolor="rgba(220,70,70,0.12)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        # shadow + line
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="rgba(0,0,0,0.06)", width=8)))
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"], mode="lines", name="Stress (Trend)",
                                 line=dict(color="rgb(220,70,70)", width=3.5)))

        # Relax CI
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list((d["relax_trend"] + d["relax_std"])) + list((d["relax_trend"] - d["relax_std"])[::-1]),
            fill="toself", fillcolor="rgba(70,170,70,0.12)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="rgba(0,0,0,0.06)", width=8)))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"], mode="lines", name="Entspannung (Trend)",
                                 line=dict(color="rgb(70,170,70)", width=3.5)))

        fig.update_layout(height=640, margin=dict(l=40, r=20, t=60, b=60),
                          xaxis=dict(type="category", tickangle=45, title="Datum"), yaxis=dict(title="Index"),
                          title="Stress- und Entspannungs-Trend")
        return fig

    if kind == "bands":
        d = df.copy()
        for c in ["delta", "theta", "alpha", "beta", "gamma"]:
            d[f"{c}_trend"] = roll_mean(d[c], smooth)
            d[f"{c}_std"] = roll_std(d[c], smooth).fillna(0)

        palette = {
            "delta_trend": ("Delta", "rgb(100,149,237)"),
            "theta_trend": ("Theta", "rgb(128,0,128)"),
            "alpha_trend": ("Alpha", "rgb(34,139,34)"),
            "beta_trend":  ("Beta",  "rgb(255,165,0)"),
            "gamma_trend": ("Gamma", "rgb(220,20,60)")
        }

        for key, (label, color) in palette.items():
            base = key.replace("_trend", "")
            ci_up = (d[key] + d[f"{base}_std"]).tolist()
            ci_dn = (d[key] - d[f"{base}_std"]).tolist()[::-1]
            fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=ci_up + ci_dn,
                                     fill="toself", fillcolor="rgba(150,150,150,0.08)", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=d[key], mode="lines", showlegend=False,
                                     line=dict(color="rgba(0,0,0,0.06)", width=6), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=d[key], mode="lines", name=label,
                                     line=dict(color=color, width=3.2)))

        fig.update_layout(height=640, margin=dict(l=40, r=20, t=60, b=60),
                          xaxis=dict(type="category", tickangle=45, title="Datum"),
                          yaxis=dict(title="Relativer Anteil", range=[0,1]),
                          title="EEG-Bänder (Trendlinien)")
        return fig

    raise ValueError("Unknown kind")

# ---------------- Matplotlib-Fallback-Rendering ----------------
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
        ax.plot(x, d["stress_trend"], c=(0.86, 0.27, 0.27), lw=3.5, zorder=3, label="Stress (Trend)")
        ax.plot(x, d["relax_trend"], c=(0.27, 0.67, 0.27), lw=3.5, zorder=3, label="Entspannung (Trend)")
        ax.legend(loc="best"); ax.set_ylabel("Index"); ax.set_title("Stress- und Entspannungs-Trend")

    elif kind == "bands":
        d = df.copy()
        for c in ["delta", "theta", "alpha", "beta", "gamma"]:
            d[f"{c}_trend"] = roll_mean(d[c], smooth)
            d[f"{c}_std"] = roll_std(d[c], smooth).fillna(0)
        series = [
            ("Delta", d["delta_trend"], d["delta_std"], (0.39, 0.58, 0.93)),
            ("Theta", d["theta_trend"], d["theta_std"], (0.28, 0.24, 0.55)),
            ("Alpha", d["alpha_trend"], d["alpha_std"], (0.13, 0.55, 0.13)),
            ("Beta",  d["beta_trend"], d["beta_std"], (1.00, 0.65, 0.00)),
            ("Gamma", d["gamma_trend"], d["gamma_std"], (0.86, 0.08, 0.24)),
        ]
        for label, y, std, col in series:
            ax.fill_between(x, y-std, y+std, alpha=0.12, color=col)
            ax.plot(x, y, lw=3, c=col, zorder=3, label=label)
        ax.set_ylim(0, 1); ax.set_ylabel("Relativer Anteil")
        ax.set_title("EEG-Bänder (Trendlinien)"); ax.legend(loc="best")

    ax.set_xticks(x); ax.set_xticklabels(xticks, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    return outpath

# ---------------- PNG->JPG Konvertierung ----------------
def convert_png_to_jpg(png_path: str, jpg_path: str, quality: int = 80, bg_color=(255,255,255)):
    if not HAS_PIL:
        raise RuntimeError("Pillow nicht installiert")
    with Image.open(png_path) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, bg_color)
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.save(jpg_path, "JPEG", quality=int(np.clip(quality, 10, 100)))
    return jpg_path

def save_plotly_as_jpg(fig: go.Figure, out_base: str, width=1600, height=1200, scale=3, jpg_quality=80):
    """Plotly -> tmp PNG -> JPG (Pillow)."""
    tmp_png = f"{out_base}__tmp.png"
    out_jpg = f"{out_base}.jpg"
    try:
        # beziehe Kaleido (wenn verfügbar) automatisch; kein engine-Argument
        pio.write_image(fig, tmp_png, width=width, height=height, scale=scale)
        if not HAS_PIL:
            raise RuntimeError("Pillow nicht installiert")
        convert_png_to_jpg(tmp_png, out_jpg, quality=jpg_quality)
        try:
            os.remove(tmp_png)
        except Exception:
            pass
        return out_jpg
    except Exception:
        try:
            if os.path.isfile(tmp_png):
                os.remove(tmp_png)
        except Exception:
            pass
        raise

def save_matplotlib_then_jpg(make_png_func, out_base: str, jpg_quality=80, **kwargs):
    out_png = f"{out_base}.png"
    make_png_func(outpath=out_png, **kwargs)
    out_jpg = f"{out_base}.jpg"
    try:
        convert_png_to_jpg(out_png, out_jpg, quality=jpg_quality)
        try:
            os.remove(out_png)
        except Exception:
            pass
        return out_jpg
    except Exception:
        return out_png

# ---------------- Chart-Baukasten ----------------
def build_charts(df: pd.DataFrame, smooth: int, y_mode: str):
    charts = {}
    if len(df) == 1:
        charts["single"] = plot_single_session_interactive(df)
    else:
        charts["stress"] = plot_stress_relax(df, smooth=smooth)
        charts["bands"]  = plot_bands(df, smooth=smooth, y_mode=y_mode)
    return charts

# ---------------- UI: Upload & Params ----------------
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

st.subheader("2) Parameter / QC")
with st.expander("Hilfe zu Parametern", expanded=False):
    st.markdown("""
**Glättungsfenster (Sessions)**: 1 = keine Glättung; höhere Werte glätten stärker.  
**Sampling-Rate**: Nur für Rohdaten-Preprocessing bzw. Zeitachsen-Fallback.
""")
smooth = st.slider("Glättungsfenster (Sessions)", 1, 15, 2, 1)
y_mode = st.selectbox("Y-Achse für Bänder", ["0–1 (fix)", "Auto (zoom)", "Abweichung vom Mittelwert (%)"], index=0)
fs = st.number_input("Sampling-Rate für Preprocessing/Timeline (Hz)", value=250.0, step=1.0)
do_preproc = st.checkbox("Preprocessing (Notch+Bandpass), falls Rohdaten", value=(True and HAS_SCIPY))

csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*.csv"), recursive=True)]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all) / (1024*1024) if n_csv > 0 else 0.0
st.info(f"Gefundene CSVs: {n_csv} — Gesamtgröße: {total_mb:.1f} MB")

if "df_summary" in st.session_state and not st.session_state["df_summary"].empty:
    if st.session_state.get("last_smooth") != smooth or st.session_state.get("last_y_mode") != y_mode:
        st.session_state["charts"] = build_charts(st.session_state["df_summary"], smooth, y_mode)
        st.session_state["last_smooth"] = smooth
        st.session_state["last_y_mode"]  = y_mode

# ---------------- Auswertung ----------------
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
            df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
            st.session_state["df_summary"] = df.copy()
            st.session_state["last_smooth"] = smooth
            st.session_state["last_y_mode"]  = y_mode
            st.session_state["charts"] = build_charts(df, smooth, y_mode)
            st.success(f"{len(df)} Session(s) ausgewertet. Anzeige unten aktualisiert.")
        if failed:
            st.subheader("Übersprungene Dateien")
            for f in failed:
                st.warning(f"{f['source']}: {f['reason']}")

# ---------------- Anzeige ----------------
df_show = st.session_state.get("df_summary", pd.DataFrame())
charts  = st.session_state.get("charts", {})

if not df_show.empty:
    if len(df_show) == 1:
        st.subheader("Einzel-Session")
        fig = charts.get("single") or plot_single_session_interactive(df_show)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Zeitverlauf (Einzel-Session)")
        ss_smooth = st.slider("Glättung (Sekunden)", 0, 30, 3, 1)
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

    df_out = df_show.copy()
    df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button("Summary CSV herunterladen",
                       data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name="summary_indices.csv", mime="text/csv")

# ---------------- Export (JPG) - Auswahl: Timeline / Balken / Kombi ----------------
st.subheader("Export (JPG, Qualität 80%)")
if df_show.empty:
    st.info("Keine Auswertung im Speicher. Erst „Auswertung starten“.")
else:
    # Different UI for single vs multi
    if len(df_show) == 1:
        st.markdown("**Einzel-Session Export** — Wähle, was gerendert werden soll (Timeline = Zeitverlauf).")
        export_choice = st.selectbox("Export:", ["Timeline (Zeitverlauf)", "Balkendiagramm", "Beide (Timeline oben, Balken unten)"])
        export_btn = st.button("JPG rendern (Einzel-Session)")
        session_csv_name = df_show.iloc[0]["source"]
        session_csv_path = find_csv_by_basename(session_csv_name, workdir)
        session_base = os.path.splitext(session_csv_name)[0]
        outdir = os.path.join(workdir, "exports"); os.makedirs(outdir, exist_ok=True)
        basepath = os.path.join(outdir, session_base)
        if export_btn:
            try:
                if export_choice == "Timeline (Zeitverlauf)":
                    # timeline only
                    if session_csv_path:
                        fig = plot_single_session_timeline(session_csv_path, fs=fs, smooth_seconds=ss_smooth, y_mode=y_mode_single)
                        # save as jpg
                        if HAS_KALEIDO:
                            out_jpg = save_plotly_as_jpg(fig, basepath + "_timeline", width=1600, height=900, scale=3, jpg_quality=80)
                        else:
                            # Matplotlib fallback: create timeline via matplotlib (reuse render_single_session_bar_and_timeline_matplotlib with only timeline)
                            out_jpg = save_matplotlib_then_jpg(
                                make_png_func=render_png_matplotlib, out_base=basepath + "_timeline",
                                jpg_quality=80, df=pd.DataFrame([df_show.iloc[0]]), kind="bands", smooth=smooth
                            )
                    else:
                        raise RuntimeError("Original-CSV für Timeline nicht gefunden.")
                elif export_choice == "Balkendiagramm":
                    # bar only
                    fig_bar = plot_single_session_interactive(df_show)
                    if HAS_KALEIDO:
                        out_jpg = save_plotly_as_jpg(fig_bar, basepath + "_bar", width=1200, height=800, scale=3, jpg_quality=80)
                    else:
                        # create small matplotlib bar
                        def make_bar_png(outpath="tmp.png", df_local=df_show):
                            plt.style.use("seaborn-v0_8-darkgrid")
                            vals = {"Stress": df_local["stress"].iloc[0], "Entspannung": df_local["relax"].iloc[0],
                                    "Delta": df_local["delta"].iloc[0], "Theta": df_local["theta"].iloc[0],
                                    "Alpha": df_local["alpha"].iloc[0], "Beta": df_local["beta"].iloc[0],
                                    "Gamma": df_local["gamma"].iloc[0]}
                            data = pd.DataFrame({"Metrik": list(vals.keys()), "Wert": list(vals.values())})
                            figm, ax = plt.subplots(figsize=(10,6), dpi=110)
                            bars = ax.bar(data["Metrik"], data["Wert"])
                            for b, h in zip(bars, data["Wert"]):
                                ax.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom")
                            figm.tight_layout()
                            figm.savefig(outpath, bbox_inches="tight")
                            plt.close(figm)
                            return outpath
                        out_jpg = save_matplotlib_then_jpg(make_bar_png, basepath + "_bar", jpg_quality=80)
                else:
                    # Both: Timeline above, bar below
                    if not session_csv_path:
                        raise RuntimeError("Original-CSV für Timeline nicht gefunden.")
                    # build combined plotly figure
                    from copy import deepcopy
                    fig_time = plot_single_session_timeline(session_csv_path, fs=fs, smooth_seconds=ss_smooth, y_mode=y_mode_single)
                    fig_bar = plot_single_session_interactive(df_show)
                    combo = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35], vertical_spacing=0.06, subplot_titles=("Zeitverlauf", "Balkendiagramm"))
                    for tr in fig_time.data:
                        combo.add_trace(deepcopy(tr), row=1, col=1)
                    for tr in fig_bar.data:
                        combo.add_trace(deepcopy(tr), row=2, col=1)
                    combo.update_layout(height=1200, margin=dict(l=40, r=20, t=60, b=60))
                    if HAS_KALEIDO:
                        out_jpg = save_plotly_as_jpg(combo, basepath + "_combo", width=1600, height=1200, scale=3, jpg_quality=80)
                    else:
                        # Matplotlib fallback: use custom function to render both (timeline + bar)
                        out_jpg = save_matplotlib_then_jpg(
                            make_png_func=render_single_session_bar_and_timeline_matplotlib,
                            out_base=basepath + "_combo",
                            jpg_quality=80,
                            csv_path=(session_csv_path if session_csv_path else ""),
                            row=df_show.iloc[0],
                            fs=fs,
                            smooth_seconds=ss_smooth,
                            y_mode=y_mode_single
                        )
                st.session_state["render_path"] = out_jpg
                st.success(f"JPG erzeugt: {out_jpg}")
            except Exception as e:
                st.error(f"Export fehlgeschlagen: {e}")
    else:
        # Multi-session export: allow rendering Stress/Entspannung or Bänder (individuell)
        st.markdown("**Multi-Session Export** — Wähle ein Motiv.")
        render_kind = st.selectbox("Motiv", ["Stress/Entspannung (Trend)", "Bänder (Trend)"], key="render_kind")
        render_btn  = st.button("JPG rendern (Multi-Session)")
        outdir = os.path.join(workdir, "exports"); os.makedirs(outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"summary_{ts}")
        if render_btn:
            kind = "stress_relax" if "Stress" in render_kind else "bands"
            try:
                if HAS_KALEIDO:
                    fig = make_beauty_figure(df_show, kind=kind, smooth=smooth)
                    out_jpg = save_plotly_as_jpg(fig, base, width=1600, height=900, scale=3, jpg_quality=80)
                else:
                    out_jpg = save_matplotlib_then_jpg(
                        make_png_func=render_png_matplotlib,
                        out_base=base,
                        jpg_quality=80,
                        df=df_show, kind=kind, smooth=smooth
                    )
                st.session_state["render_path"] = out_jpg
                st.success(f"JPG erzeugt: {out_jpg}")
            except Exception as e:
                st.error(f"Export fehlgeschlagen: {e}")

# ---------------- Vorschau + Download ----------------
if st.session_state.get("render_path") and os.path.isfile(st.session_state["render_path"]):
    p = st.session_state["render_path"]
    st.image(p, caption="Rendering-Vorschau (JPG)", use_container_width=True)
    # download name: preserve session base when single-session
    download_name = os.path.basename(p)
    if not download_name.lower().endswith(".jpg"):
        download_name = os.path.splitext(download_name)[0] + ".jpg"
    with open(p, "rb") as f:
        st.download_button("JPG herunterladen", f, file_name=download_name, mime="image/jpeg")

# ---------------- Wartung ----------------
with st.expander("Debug / Wartung", expanded=False):
    if st.button("Arbeitsordner leeren"):
        try:
            shutil.rmtree(st.session_state["workdir"])
        except Exception:
            pass
        for k in ["workdir", "df_summary", "charts", "render_path", "last_smooth", "last_y_mode"]:
            st.session_state.pop(k, None)
        st.success("Arbeitsordner geleert. Seite neu laden.")
