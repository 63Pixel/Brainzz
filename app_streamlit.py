#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# EEG-Auswertung – simplified UI: no Y-Achse option, no Sampling-Rate/preprocessing in UI
# Timeline first, then bar. JPG export (quality 80%). Downsampling presets available.
#
# Änderungen:
# - Dropbox-Download-Button: lädt beim Klick die ZIP gestreamt in workdir/downloads und bietet sie zum Download an.
# - benötigte Importe für Dropbox-Handling hinzugefügt.

import os
import re
import glob
import zipfile
import tempfile
import shutil
from datetime import datetime
from datetime import timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# neue Importe für Dropbox/HTTP-Download
import requests
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

# Pillow for PNG->JPG conversion (required for JPG export)
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# Matplotlib fallback (no Chrome required)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Basic config
st.set_page_config(page_title="EEG-Auswertung", layout="wide")
st.title("EEG-Auswertung")

# ---------- Arbeitsverzeichnis ----------
def get_workdir():
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="eeg_works_")
    return st.session_state["workdir"]

workdir = get_workdir()

# ---------- Helfer ----------
def create_test_zip(num_sessions: int = 1, rows: int = 400, fs: int = 250):
    """
    Erzeugt ein ZIP-Archiv (in-memory) mit `num_sessions` Beispiel-CSV-Dateien.
    Jede CSV ist im Dateinamen mit brainzz_YYYY-MM-DD--HH-MM-SS versehen, damit dein Parser sie erkennt.
    Rückgabe: bytes (Zip-Archiv), empfohlenes Dateiname: 'brainzz_testdata.zip'
    """
    buf = io.BytesIO()
    now = datetime.now()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(num_sessions):
            # Timestamp leicht variieren
            ts = (now + timedelta(seconds=i)).strftime("%Y-%m-%d--%H-%M-%S")
            fname = f"brainzz_{ts}_session.csv"
            # Erzeuge einfache Zeitstempel und relative Band-Spalten (Delta_1..Gamma_1)
            t = np.arange(rows) / fs
            # zufällige, aber plausible Bänder (sollten sich zu ~1 summieren)
            rng = np.random.RandomState(100 + i)
            delta = np.abs(rng.normal(loc=0.2, scale=0.03, size=rows))
            theta = np.abs(rng.normal(loc=0.15, scale=0.03, size=rows))
            alpha = np.abs(rng.normal(loc=0.35, scale=0.05, size=rows))
            beta  = np.abs(rng.normal(loc=0.2, scale=0.04, size=rows))
            gamma = np.abs(rng.normal(loc=0.1, scale=0.02, size=rows))
            # Normieren (relative Anteile)
            tot = delta + theta + alpha + beta + gamma + 1e-12
            delta /= tot; theta /= tot; alpha /= tot; beta /= tot; gamma /= tot
            # zeitspalte als ISO timestamps (relative to now)
            times = (pd.Timestamp.now() + pd.to_timedelta(np.round(t).astype(int), unit="s")).strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame({
                "datetime": times,
                "Delta_1": delta, "Theta_1": theta, "Alpha_1": alpha,
                "Beta_1": beta, "Gamma_1": gamma
            })
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            zf.writestr(fname, csv_bytes)
    buf.seek(0)
    return buf.getvalue()

PAT = re.compile(r"brainzz_(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})")

def parse_dt_from_path(path: str):
    if not path:
        return None
    m = PAT.search(path)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d--%H-%M-%S")
    except Exception:
        return None

def format_session_filename(dt: datetime):
    if not isinstance(dt, datetime):
        return None
    return f"brainzz_{dt.strftime('%Y-%m-%d--%H-%M-%S')}"

def get_mtime(path: str):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def roll_mean(s, w):
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).mean()

def roll_std(s, w):
    return s.rolling(window=max(1, int(w)), center=True, min_periods=1).std()

def recursively_extract_archives(root_dir):
    """Entpacke ZIP/SIP rekursiv (markiert extrahierte Archive)."""
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

def find_csv_by_basename(name: str, root_dir: str):
    paths = [p for p in glob.glob(os.path.join(root_dir, "**", name), recursive=True)]
    return paths[0] if paths else None

def is_good_rel(df):
    return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in ["alpha", "beta", "theta", "delta", "gamma"])

# ---------- Laden der relativen Bandanteile ----------
# wir erwarten in CSV Spalten Delta_*, Theta_*, Alpha_*, Beta_*, Gamma_* oder direkte Delta/Theta/... Spalten
def load_session_relatives(csv_path, agg="power"):
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
            # simple handling: use mean across channels
            out[b.lower()] = val.mean(axis=1)
        except Exception:
            return None
    rel = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
    tot = rel.sum(axis=1).replace(0, np.nan)
    return rel.div(tot, axis=0).dropna()

# ---------- Decimate helpers ----------
def decimate_series(y, max_points=800):
    if y is None:
        return None
    y_arr = np.asarray(y)
    n = len(y_arr)
    if n <= max_points or max_points <= 0:
        return y_arr
    idx = np.linspace(0, n-1, num=max_points, dtype=int)
    return y_arr.take(idx)

def decimate_xy(x, y, max_points=800):
    if y is None:
        return x, y
    n = len(y)
    if n <= max_points or max_points <= 0:
        return np.asarray(x), np.asarray(y)
    idx = np.linspace(0, n-1, num=max_points, dtype=int)
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return x_arr.take(idx), y_arr.take(idx)

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

def plot_stress_relax(df, smooth=1, max_points=800):
    d = df.copy()
    d["stress_trend"] = roll_mean(d["stress"], smooth)
    d["relax_trend"]  = roll_mean(d["relax"],  smooth)
    x = d["date_str"].values
    x_dec, stress_dec = decimate_xy(x, d["stress_trend"].values, max_points=max_points)
    _, relax_dec = decimate_xy(x, d["relax_trend"].values, max_points=max_points)
    long = pd.DataFrame({
        "date_str": list(x_dec) + list(x_dec),
        "Metrik": ["Stress (Trend)"] * len(x_dec) + ["Entspannung (Trend)"] * len(x_dec),
        "Wert": list(stress_dec) + list(relax_dec)
    })
    fig = px.line(long, x="date_str", y="Wert", color="Metrik", markers=False, height=360)
    fig.update_layout(xaxis=dict(type="category"))
    return fig

def plot_bands(df, smooth=1, max_points=800):
    # y-mode fixed to "0–1 (fix)"
    d = df.copy()
    d["stresswave"] = d["beta"] + d["gamma"]
    d["relaxwave"]  = d["alpha"] + d["theta"]
    for c in ["delta", "theta", "alpha", "beta", "gamma", "stresswave", "relaxwave"]:
        d[f"{c}_trend"] = roll_mean(d[c], smooth)
        d[f"{c}_std"] = roll_std(d[c], smooth).fillna(0)
    palette = {
        "delta_trend": "rgb(100,149,237)", "theta_trend": "rgb(128,0,128)",
        "alpha_trend": "rgb(34,139,34)", "beta_trend": "rgb(255,165,0)",
        "gamma_trend": "rgb(220,20,60)", "stresswave_trend": "rgb(220,120,60)",
        "relaxwave_trend": "rgb(60,200,180)"
    }
    x = d["date_str"].values
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
        y_trend = d[key].values
        y_std = d[f"{base}_std"].values if f"{base}_std" in d else np.zeros_like(y_trend)
        x_dec, y_trend_dec = decimate_xy(x, y_trend, max_points=max_points)
        _, y_std_dec = decimate_xy(x, y_std, max_points=max_points)
        ci_up = (y_trend_dec + y_std_dec).tolist()
        ci_dn = (y_trend_dec - y_std_dec).tolist()[::-1]
        fig.add_trace(go.Scatter(
            x=list(x_dec) + list(x_dec[::-1]),
            y=ci_up + ci_dn,
            fill="toself",
            fillcolor="rgba(150,150,150,0.08)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_dec, y=y_trend_dec, mode="lines",
            line=dict(color="rgba(0,0,0,0.06)", width=6),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_dec, y=y_trend_dec, mode="lines", name=label,
            line=dict(color=palette.get(key, "rgb(100,100,100)"), width=3.0)
        ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=40), xaxis=dict(type="category"), xaxis_title="Datum")
    fig.update_yaxes(range=[0, 1], title="Wert")
    return fig

# ---------- Einzel-Session Timeline (Plotly) ----------
# internal fallback sampling rate for time axis if no datetime column present
FALLBACK_FS = 250.0

def plot_single_session_timeline(csv_path, smooth_seconds=3, max_points=800):
    rel = load_session_relatives(csv_path)
    if rel is None or rel.empty:
        return go.Figure()
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

    if x_vals is None:
        secs = np.arange(len(rel)) / FALLBACK_FS
        x_vals = pd.to_datetime(secs, unit="s", origin=pd.Timestamp("1970-01-01"))
        is_datetime = True

    n = min(len(rel), len(x_vals))
    rel = rel.iloc[:n].copy()
    x_vals = pd.Series(x_vals).iloc[:n].reset_index(drop=True)

    w = max(1, int(round(smooth_seconds * FALLBACK_FS))) if smooth_seconds else 1
    for c in ["delta", "theta", "alpha", "beta", "gamma"]:
        if c in rel.columns:
            rel[c] = roll_mean(rel[c], w)
    rel["stresswave"] = roll_mean(rel["beta"] + rel["gamma"], w)
    rel["relaxwave"]  = roll_mean(rel["alpha"] + rel["theta"], w)

    bands_map = {
        "delta": "Delta", "theta": "Theta", "alpha": "Alpha",
        "beta": "Beta", "gamma": "Gamma",
        "stresswave": "Stress-Welle (Beta+Gamma)", "relaxwave": "Entspannungs-Welle (Alpha+Theta)"
    }

    fig = go.Figure()
    palette = {
        "Delta": "rgb(100,149,237)", "Theta": "rgb(128,0,128)", "Alpha": "rgb(34,139,34)",
        "Beta": "rgb(255,165,0)", "Gamma": "rgb(220,20,60)",
        "Stress-Welle (Beta+Gamma)": "rgb(220,120,60)", "Entspannungs-Welle (Alpha+Theta)": "rgb(60,200,180)"
    }

    for key, label in bands_map.items():
        if key not in rel:
            continue
        y = rel[key].values
        x_dec, y_dec = decimate_xy(x_vals.values, y, max_points=max_points)
        fig.add_trace(go.Scatter(x=x_dec, y=y_dec, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="rgba(0,0,0,0.06)", width=8)))
        fig.add_trace(go.Scatter(x=x_dec, y=y_dec, mode="lines", name=label,
                                 line=dict(color=palette.get(label, "gray"), width=2)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=40))
    if is_datetime:
        fig.update_xaxes(title_text="Uhrzeit", tickformat="%H:%M:%S", tickangle=45)
    else:
        fig.update_xaxes(title_text="Zeit [s]", tickangle=45)
    fig.update_yaxes(range=[0,1], title="Relativer Anteil")
    return fig

# ---------- Matplotlib Fallback Renderers (Export) ----------
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

def render_single_session_timeline_matplotlib(csv_path: str, smooth_seconds=3, outpath="timeline.png"):
    rel = load_session_relatives(csv_path)
    if rel is None or rel.empty:
        fig, ax = plt.subplots(figsize=(8,3), dpi=90)
        ax.text(0.5, 0.5, "Keine Zeitreihendaten", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
        return outpath

    try:
        df0 = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df0 = None

    time_labels = None
    if df0 is not None:
        cand = [c for c in df0.columns if str(c).lower() in
                ["timestamp", "time", "timesec", "t", "elapsed", "seconds", "secs", "ms", "millis", "datetime", "date_time", "date", "zeit", "uhrzeit", "clock"]]
        for c in cand:
            try:
                td = pd.to_datetime(df0[c], errors="coerce")
                if td.notna().any():
                    time_labels = td.dt.strftime("%H:%M:%S").tolist()
                    break
            except Exception:
                pass

    if time_labels is None:
        time_labels = [(pd.Timestamp("1970-01-01") + pd.to_timedelta(np.arange(len(rel))/FALLBACK_FS, unit="s")).strftime("%H:%M:%S")]

    n = min(len(rel), len(time_labels))
    rel = rel.iloc[:n].copy()
    time_labels = time_labels[:n]

    w = max(1, int(round(smooth_seconds * FALLBACK_FS))) if smooth_seconds else 1
    for c in ["delta","theta","alpha","beta","gamma"]:
        if c in rel.columns:
            rel[c] = roll_mean(rel[c], w)
    rel["stresswave"] = roll_mean(rel["beta"] + rel["gamma"], w)
    rel["relaxwave"]  = roll_mean(rel["alpha"] + rel["theta"], w)

    MAX_POINTS = 800
    x_idx = np.arange(n)
    if n > MAX_POINTS:
        idxs = np.linspace(0, n-1, num=MAX_POINTS, dtype=int)
        dec_x = idxs
        tick_step = max(1, int(MAX_POINTS / 8))
        tick_positions = idxs[::tick_step]
        tick_labels = [time_labels[i] for i in tick_positions]
    else:
        dec_x = x_idx
        tick_positions = x_idx[::max(1, int(n/8))] if n>8 else x_idx
        tick_labels = [time_labels[i] for i in tick_positions]

    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=90)
    series = [
        ("Delta", rel.get("delta")), ("Theta", rel.get("theta")), ("Alpha", rel.get("alpha")),
        ("Beta", rel.get("beta")), ("Gamma", rel.get("gamma")),
        ("Stress-Welle", rel.get("stresswave")), ("Relax-Welle", rel.get("relaxwave"))
    ]
    colors = {"Delta":(0.39,0.58,0.93),"Theta":(0.28,0.24,0.55),"Alpha":(0.13,0.55,0.13),
              "Beta":(1.00,0.65,0.00),"Gamma":(0.86,0.08,0.24),"Stress-Welle":(0.8,0.4,0.2),"Relax-Welle":(0.2,0.7,0.6)}
    for name, y in series:
        if y is not None:
            y_arr = np.asarray(y)
            y_dec = decimate_series(y_arr, max_points=MAX_POINTS)
            ax.plot(np.linspace(0, len(y_dec)-1, num=len(y_dec)), y_dec, lw=1.8, label=name, color=colors.get(name,(0.2,0.2,0.2)))

    if len(tick_labels) > 0:
        ax.set_xticks(np.linspace(0, len(dec_x)-1, num=len(tick_positions), dtype=int))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    if True:
        ax.set_ylim(0,1)
    ax.set_ylabel("Relativer Anteil")
    ax.legend(ncol=4, fontsize="small", loc="upper center", bbox_to_anchor=(0.5,1.22))
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    return outpath

def render_single_session_bar_and_timeline_matplotlib(csv_path: str, row: pd.Series, smooth_seconds=3, outpath="single_combo.png"):
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
        figm, axm = plt.subplots(figsize=(10,5), dpi=90)
        labels, v = list(vals.keys()), list(vals.values())
        bars = axm.bar(labels, v)
        for b, h in zip(bars, v):
            axm.text(b.get_x()+b.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom")
        figm.tight_layout(); figm.savefig(outpath, bbox_inches="tight"); plt.close(figm)
        return outpath

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
        time_labels = [(pd.Timestamp("1970-01-01") + pd.to_timedelta(np.arange(len(rel))/FALLBACK_FS, unit="s")).strftime("%H:%M:%S")]

    n = min(len(rel), len(time_labels))
    rel = rel.iloc[:n].copy()
    t_labels = time_labels[:n]

    w = max(1, int(round(smooth_seconds * FALLBACK_FS))) if smooth_seconds else 1
    for c in ["delta","theta","alpha","beta","gamma"]:
        if c in rel.columns:
            rel[c] = roll_mean(rel[c], w)
    rel["stresswave"] = roll_mean(rel["beta"] + rel["gamma"], w)
    rel["relaxwave"]  = roll_mean(rel["alpha"] + rel["theta"], w)

    MAX_POINTS = 900
    total_n = len(rel)
    if total_n > MAX_POINTS:
        idxs = np.linspace(0, total_n-1, num=MAX_POINTS, dtype=int)
        t_labels_dec = [t_labels[i] for i in idxs]
        rel_dec = rel.iloc[idxs].reset_index(drop=True)
    else:
        idxs = np.arange(total_n)
        t_labels_dec = t_labels
        rel_dec = rel.reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,9), dpi=90, gridspec_kw={"height_ratios":[1,2]})

    labels, v = list(vals.keys()), list(vals.values())
    bars = ax1.bar(labels, v)
    for b, y in zip(bars, v):
        ax1.text(b.get_x()+b.get_width()/2, y, f"{y:.2f}", ha="center", va="bottom")
    ax1.set_ylim(0, max(1.0, max(v)*1.15))
    ax1.set_title("Balkendiagramm (Einzel-Session)")

    series = [
        ("Delta", rel_dec.get("delta")), ("Theta", rel_dec.get("theta")), ("Alpha", rel_dec.get("alpha")),
        ("Beta", rel_dec.get("beta")), ("Gamma", rel_dec.get("gamma")),
        ("Stress-Welle", rel_dec.get("stresswave")), ("Relax-Welle", rel_dec.get("relaxwave"))
    ]
    colors = {"Delta":(0.39,0.58,0.93),"Theta":(0.28,0.24,0.55),"Alpha":(0.13,0.55,0.13),
              "Beta":(1.00,0.65,0.00),"Gamma":(0.86,0.08,0.24),"Stress-Welle":(0.8,0.4,0.2),"Relax-Welle":(0.2,0.7,0.6)}
    x_plot = np.arange(len(rel_dec))
    for name, y in series:
        if y is not None:
            ax2.plot(x_plot, np.asarray(y), lw=1.8, label=name, color=colors.get(name,(0.2,0.2,0.2)))

    tick_n = min(10, len(t_labels_dec))
    if tick_n > 1:
        ix = np.linspace(0, len(t_labels_dec)-1, num=tick_n, dtype=int)
        ax2.set_xticks(np.linspace(0, len(x_plot)-1, num=tick_n))
        ax2.set_xticklabels([t_labels_dec[i] for i in ix], rotation=45, ha="right")

    ax2.set_ylim(0,1)
    ax2.set_ylabel("Relativer Anteil")
    ax2.legend(ncol=3, fontsize="small", loc="upper center", bbox_to_anchor=(0.5,1.08))
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)
    return outpath

# ---------- JPG helpers ----------
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

def save_matplotlib_then_jpg(make_png_func, out_base: str, jpg_quality=80, **kwargs):
    out_png = f"{out_base}.png"
    kwargs = dict(kwargs)
    kwargs["outpath"] = out_png
    make_png_func(**kwargs)
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

# ---------- Chart-Baukasten ----------
def build_charts(df: pd.DataFrame, smooth: int, max_points: int = 800):
    charts = {}
    if len(df) == 1:
        charts["single"] = plot_single_session_interactive(df)
    else:
        charts["stress"] = plot_stress_relax(df, smooth=smooth, max_points=max_points)
        charts["bands"]  = plot_bands(df, smooth=smooth, max_points=max_points)
    return charts

# ---------- Dropbox helper ----------
def make_direct_dropbox_link(share_link: str) -> str:
    """
    Nimmt einen Dropbox-Freigabelink und liefert eine direkte Download-URL zurück.
    Beispiel: ersetzt ?dl=0 mit ?dl=1 und wandelt domain ggf. zu dl.dropboxusercontent.com.
    """
    if not share_link:
        raise ValueError("Kein Link angegeben.")
    parsed = urlparse(share_link)
    qs = parse_qs(parsed.query)
    qs["dl"] = ["1"]
    new_query = urlencode(qs, doseq=True)
    direct = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
    direct = direct.replace("www.dropbox.com", "dl.dropboxusercontent.com")
    return direct

# ---------- UI ----------
st.subheader("Datei-Upload")

# Zwei Spalten: links Upload + Presets, rechts Testdaten-Download
col_up, col_dl = st.columns([3, 1])

with col_up:
    uploads = st.file_uploader(
        "Dateien hochladen (ZIP/SIP mit CSVs oder einzelne CSVs)",
        type=["zip", "sip", "csv"],
        accept_multiple_files=True
    )

    DEFAULT_DISPLAY_MAX = 800

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

        # Anzeige-Dichte Presets (Display-Max)
        st.markdown("**Anzeige-Dichte (für interaktive Plots)**")
        preset = st.selectbox("Voreinstellung", ["sehr niedrig", "niedrig", "mittel (800)", "hoch", "sehr hoch", "maximum"], index=2,
                              help="Wähle eine Voreinstellung; du kannst unten einen genauen Wert überschreiben.")
        preset_map = {
            "sehr niedrig": 200,
            "niedrig": 400,
            "mittel (800)": DEFAULT_DISPLAY_MAX,
            "hoch": 1200,
            "sehr hoch": 2000,
            "maximum": 5000
        }
        preset_value = preset_map.get(preset, DEFAULT_DISPLAY_MAX)
        custom_val = st.number_input("Maximal anzuzeigende Punkte (Override)", min_value=50, max_value=20000, value=int(preset_value), step=50)
        display_max = int(custom_val) if custom_val is not None else int(preset_value)
        st.session_state["display_max"] = display_max

        # grüne Auswerten-Schaltfläche (einfaches CSS)
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #28a745;
            color: white;
            border: none;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("Auswertung starten", key="start_eval"):
            recursively_extract_archives(workdir)
            selected_csvs = [p for p in glob.glob(os.path.join(workdir, "**", "*.csv"), recursive=True)]
            if not selected_csvs:
                st.error("Keine CSVs gefunden.")
            else:
                tmpdir = tempfile.mkdtemp(prefix="eeg_proc_")
                rows, failed = [], []
                for cp in sorted(selected_csvs):
                    try:
                        pd.read_csv(cp, nrows=1)
                    except Exception:
                        failed.append({"source": os.path.basename(cp), "reason": "Keine CSV (evtl. ZIP)."})
                        continue
                    dt = parse_dt_from_path(cp) or parse_dt_from_path(os.path.dirname(cp))
                    if dt is None:
                        failed.append({"source": os.path.basename(cp), "reason": "Ungültiger Zeitstempel."})
                        continue
                    # Kein Preprocessing mehr – wir laden die relativen Bänder direkt
                    rel = load_session_relatives(cp)
                    if not is_good_rel(rel):
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
                    st.session_state["last_smooth"] = st.session_state.get("last_smooth", 2)
                    st.session_state["charts"] = build_charts(df, st.session_state.get("last_smooth", 2), max_points=display_max)
                    st.success(f"{len(df)} Session(s) ausgewertet. Anzeige unten aktualisiert.")
                if failed:
                    st.subheader("Übersprungene Dateien")
                    for f in failed:
                        st.warning(f"{f['source']}: {f['reason']}")

# ensure display_max default exists
if "display_max" not in st.session_state:
    st.session_state["display_max"] = DEFAULT_DISPLAY_MAX

with col_dl:
    st.markdown("**Testdaten**")
    st.caption("Lade eine ZIP mit Beispiel-CSVs herunter (zum Testen) oder lade die Demo direkt von Dropbox.")
    # Default Dropbox-Link (deinen Link als Voreinstellung)
    default_dropbox = "https://www.dropbox.com/scl/fi/ktn1683bsksuuw2vn4lcx/Demodaten.zip?rlkey=2ikg37dp7dhfdpcb332lz3osd&dl=0"
    dropbox_link = st.text_input("Dropbox-Freigabe-Link", value=default_dropbox)

    if st.button("Dropbox: Datei laden und anbieten"):
        try:
            direct = make_direct_dropbox_link(dropbox_link)
        except Exception as e:
            st.error(f"Ungültiger Dropbox-Link: {e}")
            direct = None

        if direct:
            fname = os.path.basename(urlparse(direct).path) or "demodaten.zip"
            dest_dir = os.path.join(workdir, "downloads")
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, fname)

            try:
                with st.spinner("Lade Datei von Dropbox…"):
                    # Streamed Download, schreibt in workdir (vermeidet OOM)
                    with requests.get(direct, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(dest_path, "wb") as fh:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    fh.write(chunk)
                st.success(f"Datei geladen: {fname}")
                # Biete die Datei zum Download an (liest file bytes)
                with open(dest_path, "rb") as fh:
                    file_bytes = fh.read()
                st.download_button("Testdaten herunterladen (von Dropbox)", data=file_bytes,
                                   file_name=fname, mime="application/zip")
                st.caption("Hinweis: Wenn die Datei sehr groß ist, kann Laden/Darstellen einige Zeit dauern.")
            except Exception as e:
                st.error(f"Fehler beim Laden von Dropbox: {e}")
                st.markdown(f"Alternativ: [Öffne den direkten Link im Browser]({direct})")

# ---------- Parameter / QC (vereinfacht) ----------
st.subheader("")
with st.expander("Workflow", expanded=False):
    st.markdown("""
    ### Kurz-Hilfe — Parameter

    **Anzeige-Dichte (Voreinstellung / Max. Punkte)**  
    Bestimmt, wie viele Datenpunkte in den interaktiven Diagrammen angezeigt werden.

    - **Niedrigere Werte** → schnelleres Laden und flüssigere Bedienung, weniger Details.  
    - **Höhere Werte** → mehr Detail, kann die App verlangsamen.

    **Voreinstellungen:** `sehr niedrig (200)`, `niedrig (400)`, **mittel (800)**, `hoch (1200)`, `sehr hoch (2000)`, `maximum (5000)`.

    ---

    **Glättungsfenster (Sessions)**  
    Glatte Darstellung über mehrere Sessions (Zusammenfassung).

    - `1` = keine Glättung, höhere Werte glätten stärker.  
    - Empfehlung: **2–4** für normale Daten.

    **Glättung (Sekunden) — Einzel-Session (Timeline)**  
    Glättet den zeitlichen Verlauf innerhalb einer einzelnen Session.

    - `0` = keine Glättung.  
    - Empfehlung: **2–5 Sekunden**.

    ---

    **Auswertung starten (grün)**  
    Startet die Analyse aller gefundenen CSVs im Arbeitsordner. Nach Abschluss siehst du die ausgewerteten Sessions und ggf. übersprungene Dateien.

    **Export (JPG)**  
    Erstellt ein JPG (80 % Qualität). Dateiname = Session-Name (z. B. `brainzz_2025-09-16--06-17-25`). Du kannst nur die Timeline oder Timeline + Balken exportieren.

    **Wenn die Timeline leer ist**  
    Stelle sicher, dass die CSV eine Zeitspalte enthält (z. B. `timestamp`, `time`, `datetime`) — sonst wird ein Zeitsprung-Fallback benutzt.
    """)

smooth = st.slider("Glättungsfenster (Sessions)", 1, 15, 2, 1)
st.session_state["last_smooth"] = smooth

csv_paths_all = [p for p in glob.glob(os.path.join(workdir, "**", "*.csv"), recursive=True)]
n_csv = len(csv_paths_all)
total_mb = sum(os.path.getsize(p) for p in csv_paths_all) / (1024*1024) if n_csv > 0 else 0.0
st.info(f"Gefundene CSVs: {n_csv} — Gesamtgröße: {total_mb:.1f} MB")

# Rebuild charts when params changed
if "df_summary" in st.session_state and not st.session_state["df_summary"].empty:
    if st.session_state.get("last_smooth") != smooth or st.session_state.get("display_max") != st.session_state.get("display_max"):
        st.session_state["charts"] = build_charts(st.session_state["df_summary"], smooth, max_points=st.session_state.get("display_max", DEFAULT_DISPLAY_MAX))
        st.session_state["last_smooth"] = smooth

# ---------- Anzeige (Timeline zuerst, dann Balken) ----------
df_show = st.session_state.get("df_summary", pd.DataFrame())
charts  = st.session_state.get("charts", {})

if not df_show.empty:
    if len(df_show) == 1:
        st.subheader("Einzel-Session")
        st.markdown("### Zeitverlauf (Einzel-Session)")
        ss_smooth = st.slider("Glättung (Sekunden)", 0, 30, 3, 1)
        csv_name = df_show.iloc[0]["source"]
        csv_path = find_csv_by_basename(csv_name, workdir)
        if csv_path:
            fig_ts = plot_single_session_timeline(csv_path, smooth_seconds=ss_smooth, max_points=st.session_state.get("display_max", DEFAULT_DISPLAY_MAX))
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Original-CSV für die Einzel-Session wurde nicht gefunden.")
        st.markdown("### Balkendiagramm (Einzel-Session)")
        fig_bar = charts.get("single") or plot_single_session_interactive(df_show)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(df_show.round(4))
    else:
        st.subheader("Stress/Entspannung")
        fig1 = charts.get("stress") or plot_stress_relax(df_show, smooth=smooth, max_points=st.session_state.get("display_max", DEFAULT_DISPLAY_MAX))
        st.plotly_chart(fig1, use_container_width=True)
        st.subheader("Bänder + Wellen")
        fig2 = charts.get("bands") or plot_bands(df_show, smooth=smooth, max_points=st.session_state.get("display_max", DEFAULT_DISPLAY_MAX))
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Tabelle")
        st.dataframe(df_show.round(4))

    df_out = df_show.copy()
    df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button("Summary CSV herunterladen",
                       data=df_out.to_csv(index=False).encode("utf-8"),
                       file_name="summary_indices.csv", mime="text/csv")

# ---------- Export (JPG) ----------
st.subheader("Export (JPG)")
if df_show.empty:
    st.info("Keine Auswertung im Speicher. Erst „Auswertung starten“.")
else:
    if len(df_show) == 1:
        st.markdown("**Einzel-Session Export** — Timeline ist oben, Balkendiagramm unten.")
        export_choice = st.selectbox("Export:", ["Timeline (Zeitverlauf)", "Beide (Timeline oben, Balken unten)"])
        export_btn = st.button("JPG rendern (Einzel-Session)")
        session_csv_name = df_show.iloc[0]["source"]
        session_csv_path = find_csv_by_basename(session_csv_name, workdir)
        dt = parse_dt_from_path(session_csv_path) or parse_dt_from_path(session_csv_name)
        session_base = format_session_filename(dt) if dt else os.path.splitext(session_csv_name)[0]
        outdir = os.path.join(workdir, "exports"); os.makedirs(outdir, exist_ok=True)
        basepath = os.path.join(outdir, session_base)
        if export_btn:
            try:
                if export_choice == "Timeline (Zeitverlauf)":
                    if not session_csv_path:
                        raise RuntimeError("Original-CSV für Timeline nicht gefunden.")
                    out_jpg = save_matplotlib_then_jpg(
                        make_png_func=render_single_session_timeline_matplotlib,
                        out_base=basepath + "_timeline",
                        jpg_quality=80,
                        csv_path=session_csv_path,
                        smooth_seconds=ss_smooth
                    )
                else:
                    if not session_csv_path:
                        raise RuntimeError("Original-CSV für Timeline nicht gefunden.")
                    out_jpg = save_matplotlib_then_jpg(
                        make_png_func=render_single_session_bar_and_timeline_matplotlib,
                        out_base=basepath + "_combo",
                        jpg_quality=80,
                        csv_path=session_csv_path,
                        row=df_show.iloc[0],
                        smooth_seconds=ss_smooth
                    )
                st.session_state["render_path"] = out_jpg
                st.success(f"JPG erzeugt: {out_jpg}")
            except Exception as e:
                st.error(f"Export fehlgeschlagen: {e}")
    else:
        st.markdown("**Multi-Session Export** — Wähle ein Motiv.")
        render_kind = st.selectbox("Motiv", ["Stress/Entspannung (Trend)", "Bänder (Trend)"], key="render_kind")
        render_btn  = st.button("JPG rendern (Multi-Session)")
        outdir = os.path.join(workdir, "exports"); os.makedirs(outdir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"summary_{ts}")
        if render_btn:
            kind = "stress_relax" if "Stress" in render_kind else "bands"
            try:
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

# ---------- Vorschau + Download ----------
if st.session_state.get("render_path") and os.path.isfile(st.session_state["render_path"]):
    p = st.session_state["render_path"]
    st.image(p, caption="Rendering-Vorschau (JPG)", use_container_width=True)
    download_name = os.path.basename(p)
    if not download_name.lower().endswith(".jpg"):
        download_name = os.path.splitext(download_name)[0] + ".jpg"
    with open(p, "rb") as f:
        st.download_button("JPG herunterladen", f, file_name=download_name, mime="image/jpeg")

# ---------- Wartung ----------
with st.expander("Debug / Wartung", expanded=False):
    if st.button("Arbeitsordner leeren"):
        try:
            shutil.rmtree(st.session_state["workdir"])
        except Exception:
            pass
        for k in ["workdir", "df_summary", "charts", "render_path", "last_smooth", "display_max"]:
            st.session_state.pop(k, None)
        st.success("Arbeitsordner geleert. Seite neu laden.")

# Copyright-Hinweis (zentriert, dezent)
st.markdown(
    f"<div style='text-align:center; color:#666; font-size:12px; margin-top:10px;'>© Stefan Nitz {datetime.now().year}</div>",
    unsafe_allow_html=True
)
