import os, io, zipfile, glob, re, tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ftplib import FTP

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
    rel = rel.div(total, axis=0).dropna()  # Zeilen normalisieren
    return rel

def build_session_table(root_dir: str) -> pd.DataFrame:
    csvs = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    rows = []
    for cp in sorted(csvs):
        dt = parse_dt_from_path(cp) or parse_dt_from_path(os.path.dirname(cp))
        if dt is None:
            continue
        rel = load_session_relatives(cp)
        if rel is None or rel.empty:
            continue
        alpha = rel["alpha"].mean()
        beta  = rel["beta"].mean()
        rows.append({
            "datetime": dt,
            "alpha": float(alpha),
            "beta": float(beta),
            "theta": float(rel["theta"].mean()),
            "delta": float(rel["delta"].mean()),
            "gamma": float(rel["gamma"].mean()),
            "stress": float(beta/(alpha+1e-9)),
            "relax":  float(alpha/(beta+1e-9)),
        })
    df = pd.DataFrame(rows).dropna().sort_values("datetime").reset_index(drop=True)
    return df

def rolling(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(window=win, center=True, min_periods=1).mean()

# ---------- Plotfunktionen ----------
def plot_stress_relax(df: pd.DataFrame, smooth:int=5):
    df = df.copy()
    df["date_str"] = df["datetime"].dt.strftime("%d-%m-%y")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(df.index, df["stress"], color="red", s=60, alpha=0.6, label="Stress (Sessions)")
    ax.scatter(df.index, df["relax"],  color="green", s=60, alpha=0.6, label="Entspannung (Sessions)")
    ax.plot(df.index, rolling(df["stress"], smooth), color="red", linewidth=2.5, label="Stress (Trend)")
    ax.plot(df.index, rolling(df["relax"],  smooth), color="green", linewidth=2.5, label="Entspannung (Trend)")
    uniq = df.drop_duplicates("date_str")
    ax.set_xticks(uniq.index); ax.set_xticklabels(uniq["date_str"], rotation=45, ha="right")
    ax.set_title("Stress- und Entspannungs-Index (ohne Lücken)")
    ax.set_xlabel("Datum"); ax.set_ylabel("Indexwert"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig

def plot_bands(df: pd.DataFrame, smooth:int=5, with_ci: bool=True):
    d = df.copy()
    d["stresswave"] = d["beta"] + d["gamma"]
    d["relaxwave"]  = d["alpha"] + d["theta"]
    d["date_str"] = d["datetime"].dt.strftime("%d-%m-%y")
    cols = ["delta","theta","alpha","beta","gamma","stresswave","relaxwave"]
    means = {c: rolling(d[c], smooth) for c in cols}
    stds  = {c: d[c].rolling(window=smooth, center=True, min_periods=1).std().fillna(0.0) for c in cols}
    x = np.arange(len(d))
    colors = {"delta":"blue","theta":"purple","alpha":"green","beta":"orange","gamma":"gray",
              "stresswave":"darkorange","relaxwave":"turquoise"}
    styles = {"stresswave":":","relaxwave":":"}
    fig, ax = plt.subplots(figsize=(8,4))
    for c in cols:
        m = means[c].to_numpy(dtype=float); ax.plot(x, m, linewidth=2.5, label=c.capitalize(),
                                                    color=colors[c], linestyle=styles.get(c,"-"))
        if with_ci:
            s = stds[c].to_numpy(dtype=float); lo = np.clip(m - s, 0, 1); hi = np.clip(m + s, 0, 1)
            ax.fill_between(x, lo, hi, alpha=0.15, color=colors[c])
    uniq = d.drop_duplicates("date_str")
    ax.set_xticks(uniq.index); ax.set_xticklabels(uniq["date_str"], rotation=45, ha="right")
    ax.set_title("EEG-Bandanteile mit Stress-/Entspannungswellen")
    ax.set_xlabel("Datum"); ax.set_ylabel("Relativer Anteil"); ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend(ncol=3)
    fig.tight_layout()
    return fig

def plot_single_session(df: pd.DataFrame):
    vals = {
        "Stress": df["stress"].iloc[0],
        "Entspannung": df["relax"].iloc[0],
        "Delta": df["delta"].iloc[0],
        "Theta": df["theta"].iloc[0],
        "Alpha": df["alpha"].iloc[0],
        "Beta": df["beta"].iloc[0],
        "Gamma": df["gamma"].iloc[0],
    }
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(vals.keys(), vals.values(), color=["red","green","blue","purple","green","orange","gray"])
    ax.set_title("Einzel-Session Übersicht")
    ax.set_ylabel("Wert")
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig

# ---------- Eingabe ----------
st.subheader("1) Datenquelle wählen")
mode = st.radio("Quelle", ["Datei-Upload (ZIP/Ordner als ZIP)", "FTP-Download"], horizontal=True)

workdir = tempfile.mkdtemp(prefix="eeg_works_")

if mode.startswith("Datei-Upload"):
    up = st.file_uploader("ZIP-Datei hochladen", type=["zip"])
    if up is not None:
        zbytes = up.read()
        with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
            zf.extractall(workdir)
        st.success("ZIP entpackt.")
elif mode.startswith("FTP"):
    st.info("FTP nur unverschlüsselt (Standard-FTP).")
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

st.subheader("2) Parameter")
smooth = st.slider("Glättungsfenster (Sessions)", min_value=3, max_value=11, value=5, step=2)

if st.button("Auswertung starten"):
    inner_zips = [os.path.join(workdir, f) for f in os.listdir(workdir) if f.lower().endswith(".zip")]
    for zp in inner_zips:
        name = os.path.splitext(os.path.basename(zp))[0]
        out = os.path.join(workdir, name); os.makedirs(out, exist_ok=True)
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(out)
        except zipfile.BadZipFile:
            pass

    df = build_session_table(workdir)
    if df.empty:
        st.error("Keine gültigen Sessions gefunden.")
    else:
        if len(df) == 1:
            st.subheader("Einzel-Session Analyse")
            st.pyplot(plot_single_session(df), clear_figure=True, use_container_width=True)
            st.dataframe(df.round(3))
        else:
            st.subheader("Stress/Entspannung (ohne Lücken)")
            st.pyplot(plot_stress_relax(df, smooth=smooth), clear_figure=True, use_container_width=True)
            st.subheader("Bänder + Stress-/Entspannungswellen")
            st.pyplot(plot_bands(df, smooth=smooth, with_ci=True), clear_figure=True, use_container_width=True)

        # Export
        df_out = df.copy(); df_out["date_str"] = df_out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.download_button("Summary CSV herunterladen", data=df_out.to_csv(index=False).encode("utf-8"),
                           file_name="summary_indices.csv", mime="text/csv")
