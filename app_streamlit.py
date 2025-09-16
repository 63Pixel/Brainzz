#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EEG-Auswertung – Streamlit-App
#
# Features:
# - JPG-Export (Qualität 80%)
# - Dateiname beim Download = Session-Basisname
# - Zeitachse zeigt Uhrzeit aus CSV (wenn vorhanden)
# - Schattierung + sparse markers für bessere Lesbarkeit
# - Kaleido ohne 'engine' (Deprecation fix); Matplotlib-Fallback

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

# optional: Plotly→Kaleido
try:
    import kaleido  # noqa: F401
    HAS_KALEIDO = True
except Exception:
    HAS_KALEIDO = False

# optional: Pillow für PNG->JPG Konvertierung
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
st.caption("Upload (ZIP/SIP/CSV) → Auswertung → JPG-Export (Qualität 80%)")


# ---------- Arbeitsverzeichnis ----------
def get_workdir():
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="eeg_works_")
    return st.session_state["workdir"]

workdir = get_workdir()


# ---------- Helfer ----------
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
    """Filter nur, wenn keine fertigen Bandspalten vorliegen."""
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
    """Lädt relative Bandanteile (Delta..Gamma) zeitaufgelöst.
       Erwartet Spalten Delta_*, Theta_*, Alpha_*, Beta_*, Gamma_* oder einzelne Delta/Theta/... Spalten.
       Rückgabe: DataFrame ['delta','theta','alpha','beta','gamma'] (0..1) oder None.
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
                val = val.pow(2)
            out[b.lower()] = val.mean(axis=1)
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

def roll_mean(s, w):
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).mean()

def roll_std(s, w):
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).std()

def find_csv_by_basename(name: str, root_dir: str):
    paths = [p for p in glob.glob(os.path.join(root_dir, "**", name), recursive=True)]
    return paths[0] if paths else None


# ---------- Plot-Funktionen ----------
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


# ---------- Einzel-Session: Zeitachse als Uhrzeit (sparse markers) ----------
def plot_single_session_timeline(csv_path, fs=250.0, smooth_seconds=3, y_mode="0–1 (fix)"):
    rel = load_session_relatives(csv_path)
    if rel is None or rel.empty:
        return go.Figure()

    # CSV laden (versuchen Zeitstempel zu finden)
    try:
        df0 = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df0 = None

    x_vals = None
    is_datetime = False
    if df0 is not None:
        cand = [c for c in df0.columns if str(c).lower() in
                ["timestamp", "time", "timesec", "t", "elapsed", "seconds", "secs",
                 "ms", "millis", "datetime", "date_time", "date", "zeit", "clock", "uhrzeit"]]
        for c in cand:
            s = df0[c]
            if np.issubdtype(s.dtype, np.number):
                val = pd.to_numeric(s, errors="coerce").astype(float)
                if np.nanmax(val) > 1e6:
                    # treat as ms epoch or ms elapsed -> to datetime
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

    # fallback: use relative seconds converted to datetime origin for prettier labels
    if x_vals is None:
        secs = np.arange(len(rel)) / (fs if fs and fs > 0 else 1.0)
        x_vals = pd.to_datetime(secs, unit="s", origin=pd.Timestamp("1970-01-01"))
        is_datetime = True

    # angleichen
    n = min(len(rel), len(x_vals))
    rel = rel.iloc[:n].copy()
    x_vals = pd.Series(x_vals).iloc[:n].reset_index(drop=True)

    # glätten
    w = max(1, int(round((smooth_seconds if smooth_seconds else 0) * (fs if fs else 1))))
    for c in ["delta", "theta", "alpha", "beta", "gamma"]:
        if c in rel.columns:
            rel[c] = roll_mean(rel[c], w)
    rel["stresswave"] = roll_mean(rel["beta"] + rel["gamma"], w)
    rel["relaxwave"]  = roll_mean(rel["alpha"] + rel["theta"], w)

    # prepare per-band traces with sparse markers
    bands_map = {
        "delta": "Delta", "theta": "Theta", "alpha": "Alpha",
        "beta": "Beta", "gamma": "Gamma",
        "stresswave": "Stress-Welle (Beta+Gamma)",
        "relaxwave": "Entspannungs-Welle (Alpha+Theta)"
    }

    fig = go.Figure()
    total_points = len(rel)
    marker_step = max(1, int(total_points / 80))  # ~80 marker points per series

    for key, label in bands_map.items():
        if key not in rel.columns:
            continue
        y = rel[key].values
        # faint shadow behind line first (for depth)
        fig.add_trace(go.Scatter(
            x=x_vals, y=y, mode="lines",
            showlegend=False, hoverinfo="skip",
            line=dict(color="rgba(0,0,0,0.06)", width=8)
        ))
        # main line
        fig.add_trace(go.Scatter(
            x=x_vals, y=y, mode="lines",
            name=label, line=dict(width=2)
        ))
        # sparse markers
        idx = np.arange(0, total_points, marker_step)
        fig.add_trace(go.Scatter(
            x=x_vals.iloc[idx], y=y[idx],
            mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=6, opacity=0.22)
        ))

    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=40))
    if is_datetime:
        fig.update_xaxes(title_text="Uhrzeit", tickformat="%H:%M:%S", tickangle=45)
    else:
        fig.update_xaxes(title_text="Zeit [s]", tickangle=45)
    if y_mode == "0–1 (fix)":
        fig.update_yaxes(range=[0, 1], title="Relativer Anteil")
    else:
        fig.update_yaxes(title="Relativer Anteil")
    return fig


# ---------- hübsche Plotly-Renderings mit Schattierung & sparse markers ----------
def make_beauty_figure(df, kind="stress_relax", smooth=1):
    x = df["date_str"]
    fig = go.Figure()

    if kind == "stress_relax":
        d = df.copy()
        d["stress_trend"] = roll_mean(d["stress"], smooth)
        d["relax_trend"]  = roll_mean(d["relax"],  smooth)
        d["stress_std"]   = roll_std(d["stress"], smooth).fillna(0)
        d["relax_std"]    = roll_std(d["relax"],  smooth).fillna(0)

        # stress band (area)
        fig.add_trace(go.Scatter(x=x, y=(d["stress_trend"] + d["stress_std"]).tolist(),
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=(d["stress_trend"] - d["stress_std"]).tolist(),
                                 fill="tonexty", fillcolor="rgba(220,70,70,0.18)",
                                 line=dict(width=0), name="Stress Band", hoverinfo="skip"))

        # shadow behind main line
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"], mode="lines",
                                 line=dict(color="rgba(0,0,0,0.06)", width=8),
                                 hoverinfo="skip", showlegend=False))
        # main + sparse markers
        total = len(d)
        step = max(1, int(total/60))
        fig.add_trace(go.Scatter(x=x, y=d["stress_trend"], name="Stress (Trend)",
                                 mode="lines", line=dict(color="rgb(220,70,70)", width=3.5)))
        fig.add_trace(go.Scatter(x=x.iloc[::step], y=d["stress_trend"].iloc[::step],
                                 mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=7, color="rgb(220,70,70)", opacity=0.25)))

        # relax band
        fig.add_trace(go.Scatter(x=x, y=(d["relax_trend"] + d["relax_std"]).tolist(),
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=(d["relax_trend"] - d["relax_std"]).tolist(),
                                 fill="tonexty", fillcolor="rgba(70,170,70,0.18)",
                                 line=dict(width=0), name="Entspannung Band", hoverinfo="skip"))

        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"], mode="lines",
                                 line=dict(color="rgba(0,0,0,0.06)", width=8),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=d["relax_trend"], name="Entspannung (Trend)",
                                 mode="lines", line=dict(color="rgb(70,170,70)", width=3.5)))
        fig.add_trace(go.Scatter(x=x.iloc[::step], y=d["relax_trend"].iloc[::step],
                                 mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=7, color="rgb(70,170,70)", opacity=0.25)))

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
            "gamma_trend": ("Gamma", "rgb(220,20,60)")
        }

        total = len(d)
        step = max(1, int(total/60))

        for key, (label, color) in palette.items():
            if key not in d:
                continue
            # subtle shadow
            fig.add_trace(go.Scatter(x=x, y=d[key], mode="lines",
                                     line=dict(color="rgba(0,0,0,0.06)", width=6),
                                     hoverinfo="skip", showlegend=False))
            # main line
            fig.add_trace(go.Scatter(x=x, y=d[key], name=label, mode="lines",
                                     line=dict(color=color, width=3.5)))
            # sparse markers
            fig.add_trace(go.Scatter(x=x.iloc[::step], y=d[key].iloc[::step],
                                     mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=7, color=color, opacity=0.25)))

        fig.update_layout(height=640, margin=dict(l=40, r=20, t=60, b=60),
                          title="EEG-Bänder (Trendlinien)",
                          xaxis=dict(type="category", tickangle=45, title="Datum"),
                          yaxis=dict(title="Relativer Anteil", range=[0, 1]))
        return fig

    raise ValueError("Unknown kind")


# ---------- Matplotlib-Fallback-Renderer ----------
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
        ax.scatter(x, d["stress_trend"], s=80, c=(0.86, 0.27, 0.27), zorder=4)
        ax.plot(x, d["relax_trend"], c=(0.27, 0.67, 0.27), lw=3.5, zorder=3, label="Entspannung (Trend)")
        ax.scatter(x, d["relax_trend"], s=80, c=(0.27, 0.67, 0.27), zorder=4)
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
            ax.plot(np.arange(len(y)), y, lw=3, c=col, zorder=3, label=label)
            ax.scatter(np.arange(len(y)), y, s=64, c=[col], zorder=4)
        ax.set_ylim(0, 1); ax.set_ylabel("Relativer Anteil")
        ax.set_title("EEG-Bänder (Trendlinien)"); ax.legend(loc="best")

    ax.set_xticks(x); ax.set_xticklabels(xticks, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    return outpath


def render_single_session_bar_and_timeline_matplotlib(csv_path: str, row: pd.Series,
                                                      fs=250.0, smooth_seconds=3, y_mode="0–1 (fix)",
                                                      outpath="single_combo.png"):
    vals = {
        "Stress": float(row["stress"]),
        "Entspannung": float(row["relax"]),
        "Delta": float(row["delta"]),
        "Theta": float(row["theta"]),
        "Alpha": float(row["alpha"]),
        "Beta": float(row["beta"]),
        "Gamma": float(row["gamma"]),
    }

    rel = load_session_relatives(csv_path) if csv_path else None
    if rel is None or rel.empty:
        fig, ax = plt.subplots(figsize=(16, 10), dpi=110)
        labels, v = list(vals.keys()), list(vals.values())
        bars = ax.bar(labels, v)
        for b, y in zip(bars, v):
            ax.text(b.get_x()+b.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, max(1.0, max(v)*1.15))
        ax.set_title("Einzel-Session – Balkendiagramm")
        fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
        return outpath

    # try to extract time labels
    try:
        df0 = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df0 = None
    time_labels = None
    if df0 is not None:
        cand = [c for c in df0.columns if str(c).lower() in
                ["timestamp","time","datetime","date_time","date","uhrzeit","zeit","clock"]]
        for c in cand:
            try:
                td = pd.to_datetime(df0[c], errors="coerce")
                if td.notna().any():
                    time_labels = td.dt.strftime("%H:%M:%S").tolist()
                    break
            except Exception:
                pass
    if time_labels is None:
        time_labels = [(pd.Timestamp("1970-01-01") + pd.to_timedelta(np.arange(len(rel))/fs, unit="s")).strftime("%H:%M:%S")]

    n = min(len(rel), len(time_labels))
    rel = rel.iloc[:n].copy()
    t = np.arange(n)

    w = max(1, int(round((smooth_seconds if smooth_seconds else 0) * (fs if fs else 1))))
    for c in ["delta","theta","alpha","beta","gamma"]:
        if c in rel.columns:
            rel[c] = roll_mean(rel[c], w)
    rel["stresswave"] = roll_mean(rel["beta"] + rel["gamma"], w)
    rel["relaxwave"]  = roll_mean(rel["alpha"] + rel["theta"], w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), dpi=110, gridspec_kw={"height_ratios":[1,2]})

    labels, v = list(vals.keys()), list(vals.values())
    bars = ax1.bar(labels, v)
    for b, y in zip(bars, v):
        ax1.text(b.get_x()+b.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
    ax1.set_ylim(0, max(1.0, max(v)*1.15))
    ax1.set_title("Einzel-Session – Balkendiagramm")

    series = [
        ("Delta",      rel.get("delta")),
        ("Theta",      rel.get("theta")),
        ("Alpha",      rel.get("alpha")),
        ("Beta",       rel.get("beta")),
        ("Gamma",      rel.get("gamma")),
        ("Stress-Welle (Beta+Gamma)",  rel.get("stresswave")),
        ("Entspannungs-Welle (Alpha+Theta)", rel.get("relaxwave")),
    ]
    for name, y in series:
        if y is not None:
            ax2.plot(t, y, lw=2.0, label=name)
            ax2.scatter(t, y, s=36, alpha=0.6)
    ax2.set_xlabel("Uhrzeit")
    ax2.set_xticks(t)
    ax2.set_xticklabels(time_labels[:n], rotation=45)
    ax2.set_ylabel("Relativer Anteil")
    if y_mode == "0–1 (fix)":
        ax2.set_ylim(0, 1)
    ax2.legend(loc="best")
    ax2.set_title("Einzel-Session – Zeitverlauf")

    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    return outpath


# ---------- PNG->JPG Konvertierung (Pillow) ----------
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


# ---------- Plotly save -> PNG -> JPG (ohne engine=) ----------
def save_plotly_as_jpg(fig: go.Figure, out_base: str, width=1600, height=1200, scale=3, jpg_quality=80):
    """Speichert Plotly-Figur: write_image -> tmp PNG -> JPG (Pillow)."""
    tmp_png = f"{out_base}__tmp.png"
    out_jpg = f"{out_base}.jpg"
    try:
        # write_image ohne engine argument (kaleido wird verwendet, falls vorhanden)
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
        # fallback: keep PNG if conversion fails
        return out_png


# ---------- Chart-Baukasten ----------
def build_charts(df: pd.DataFrame, smooth: int, y_mode: str):
    charts = {}
    if len(df) == 1:
        charts["single"] = plot_single_session_interactive(df)
    else:
        charts["stress"] = plot_stress_relax(df, smooth=smooth)
        charts["bands"]  = plot_bands(df, smooth=smooth, y_mode=y_mode)
    return charts


# ---------- UI: Upload & Params ----------
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


# ---------- Auswertung ----------
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


# ---------- Anzeige ----------
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


# ---------- Export (nur JPG, Qualität 80%) ----------
st.subheader("Export (JPG, Qualität 80%)")
if df_show.empty:
    st.info("Keine Auswertung im Speicher. Erst „Auswertung starten“.")
else:
    if len(df_show) == 1:
        st.markdown("**Einzel-Session-Export (Balkendiagramm + Zeitverlauf)**")
        exp_smooth = st.slider("Glättung (Sekunden) für Export", 0, 30, 3, 1, key="exp_smooth_single")
        exp_y = st.selectbox("Y-Achse (Export)", ["0–1 (fix)", "Auto (zoom)"], index=0, key="exp_y_single")
        render_btn_single = st.button("JPG rendern", key="render_btn_single")
        st.session_state.setdefault("render_path", "")

        if render_btn_single:
            outdir = os.path.join(workdir, "exports"); os.makedirs(outdir, exist_ok=True)
            csv_name = df_show.iloc[0]["source"]
            sessionbase = os.path.splitext(csv_name)[0]
            base = os.path.join(outdir, sessionbase)
            csv_path = find_csv_by_basename(csv_name, workdir)
            try:
                # Prefer Plotly + Kaleido, fallback to Matplotlib if any error
                if HAS_KALEIDO and csv_path:
                    from copy import deepcopy
                    # top: bar
                    fig_bar = plot_single_session_interactive(df_show)
                    # bottom: timeline
                    fig_time = plot_single_session_timeline(csv_path, fs=fs, smooth_seconds=exp_smooth, y_mode=exp_y)
                    combo = make_subplots(rows=2, cols=1, shared_xaxes=False,
                                          row_heights=[0.35, 0.65], vertical_spacing=0.08,
                                          subplot_titles=("Balkendiagramm", "Zeitverlauf"))
                    for tr in fig_bar.data:
                        combo.add_trace(deepcopy(tr), row=1, col=1)
                    for tr in fig_time.data:
                        combo.add_trace(deepcopy(tr), row=2, col=1)
                    combo.update_layout(height=1200, margin=dict(l=40, r=20, t=60, b=60))
                    out_path = save_plotly_as_jpg(combo, base, width=1600, height=1200, scale=3, jpg_quality=80)
                else:
                    out_path = save_matplotlib_then_jpg(
                        make_png_func=render_single_session_bar_and_timeline_matplotlib,
                        out_base=base,
                        jpg_quality=80,
                        csv_path=(csv_path if csv_path else ""),
                        row=df_show.iloc[0],
                        fs=fs,
                        smooth_seconds=exp_smooth,
                        y_mode=exp_y
                    )
                st.session_state["render_path"] = out_path
                st.success(f"Bild erzeugt: {out_path}")
            except Exception as e:
                # fallback: try Matplotlib if Plotly/Kaleido failed
                try:
                    out_path = save_matplotlib_then_jpg(
                        make_png_func=render_single_session_bar_and_timeline_matplotlib,
                        out_base=base,
                        jpg_quality=80,
                        csv_path=(csv_path if csv_path else ""),
                        row=df_show.iloc[0],
                        fs=fs,
                        smooth_seconds=exp_smooth,
                        y_mode=exp_y
                    )
                    st.session_state["render_path"] = out_path
                    st.success(f"Bild erzeugt (Fallback Matplotlib): {out_path}")
                except Exception as e2:
                    st.error(f"Export fehlgeschlagen: {e}; Fallback fehlgeschlagen: {e2}")

    else:
        render_kind = st.selectbox("Motiv", ["Stress/Entspannung (Trend)", "Bänder (Trend)"], key="render_kind")
        render_btn  = st.button("JPG rendern", key="render_btn")
        st.session_state.setdefault("render_path", "")
        if render_btn:
            kind = "stress_relax" if "Stress" in render_kind else "bands"
            outdir = os.path.join(workdir, "exports"); os.makedirs(outdir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.join(outdir, f"summary_{ts}")
            try:
                if HAS_KALEIDO:
                    fig = make_beauty_figure(df_show, kind=kind, smooth=smooth)
                    out_path = save_plotly_as_jpg(fig, base, width=1600, height=900, scale=3, jpg_quality=80)
                else:
                    out_path = save_matplotlib_then_jpg(
                        make_png_func=render_png_matplotlib,
                        out_base=base,
                        jpg_quality=80,
                        df=df_show, kind=kind, smooth=smooth
                    )
                st.session_state["render_path"] = out_path
                st.success(f"Bild erzeugt: {out_path}")
            except Exception as e:
                try:
                    out_path = save_matplotlib_then_jpg(
                        make_png_func=render_png_matplotlib,
                        out_base=base,
                        jpg_quality=80,
                        df=df_show, kind=kind, smooth=smooth
                    )
                    st.session_state["render_path"] = out_path
                    st.success(f"Bild erzeugt (Fallback Matplotlib): {out_path}")
                except Exception as e2:
                    st.error(f"Export fehlgeschlagen: {e}; Fallback fehlgeschlagen: {e2}")


# Vorschau + Download
if st.session_state.get("render_path") and os.path.isfile(st.session_state["render_path"]):
    p = st.session_state["render_path"]
    st.image(p, caption="Rendering-Vorschau", use_container_width=True)
    # Set download name: for single session, we used sessionbase; otherwise keep filename but ensure .jpg
    download_name = os.path.basename(p)
    if not download_name.lower().endswith(".jpg"):
        download_name = os.path.splitext(download_name)[0] + ".jpg"
    with open(p, "rb") as f:
        st.download_button("JPG herunterladen", f, file_name=download_name, mime="image/jpeg")


# Wartung
with st.expander("Debug / Wartung", expanded=False):
    if st.button("Arbeitsordner leeren"):
        try:
            shutil.rmtree(st.session_state["workdir"])
        except Exception:
            pass
        for k in ["workdir", "df_summary", "charts", "render_path", "last_smooth", "last_y_mode"]:
            st.session_state.pop(k, None)
        st.success("Arbeitsordner geleert. Seite neu laden.")
