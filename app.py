import time
from dataclasses import asdict
from io import BytesIO
from typing import Tuple

import pandas as pd
import streamlit as st

import main as core
from utils import DetectionConfig, generate_sample_dataset, get_logger, risk_level, to_download_bytes

st.set_page_config(
    page_title="Silent Guardian — Unsupervised Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.05), transparent 25%),
                          radial-gradient(circle at 85% 30%, rgba(239, 68, 68, 0.05), transparent 25%);
    }

    /* Glassmorphism for sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glowing primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        border: none;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.6);
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
    }

    /* Standard buttons hover effects */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: #38bdf8;
        color: #38bdf8;
    }

    /* Headers and Titles */
    h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Metric cards (KPIs) hover animations */
    [data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
    }

    /* Animated Tabs */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)


def color_anomaly_rows(row: pd.Series):
    is_anom = int(row.get("final_prediction", 1)) == -1
    if is_anom:
        # Stronger, more obvious anomaly highlight.
        return ["background-color: #ff3b30; color: #ffffff; font-weight: 600;"] * len(row)
    return [""] * len(row)


def render_kpis(results: pd.DataFrame):
    total = len(results)
    anom = int((results["final_prediction"] == -1).sum())
    normal = total - anom
    avg_risk = float(results["hybrid_score"].mean()) if total else 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total events", f"{total}")
    col2.metric("Anomalies detected", f"{anom}")
    col3.metric("Normal events", f"{normal}")
    col4.metric("Avg risk score", f"{avg_risk:.3f}", help="Average of final score (0-1)")


def make_cfg(contamination: float, threshold: float, mode: str, ae_epochs: int, ae_batch: int) -> DetectionConfig:
    return DetectionConfig(
        contamination=float(contamination),
        threshold=float(threshold),
        model_mode=str(mode),
        ae_epochs=int(ae_epochs),
        ae_batch_size=int(ae_batch),
    )


@st.cache_data(show_spinner=False)
def run_detection_cached(df: pd.DataFrame, cfg: DetectionConfig) -> Tuple[pd.DataFrame, dict]:
    return core.detect_anomalies(df, cfg)


@st.cache_data(show_spinner=False)
def generate_sample_dataset_cached(n_rows: int, attack_ratio: float, seed: int) -> pd.DataFrame:
    return generate_sample_dataset(n_rows=n_rows, attack_ratio=attack_ratio, seed=seed)


def _cfg_key(cfg: DetectionConfig) -> tuple:
    d = asdict(cfg)
    return (
        d["model_mode"],
        float(d["contamination"]),
        float(d["threshold"]),
        int(d["ae_epochs"]),
        int(d["ae_batch_size"]),
        float(d["ae_validation_split"]),
        int(d["random_state"]),
    )


def _init_state():
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("df_source", None)
    st.session_state.setdefault("upload_sig", None)  # (name, size) — only reset on NEW upload
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("meta", None)
    st.session_state.setdefault("last_cfg_key", None)
    st.session_state.setdefault("sim_running", False)
    st.session_state.setdefault("sim_idx", 0)


def read_csv_robust(uploaded_file) -> pd.DataFrame:
    """
    Kaggle/Windows CSVs are often cp1252/latin1 (e.g. byte 0x92). Try a few common encodings.
    """
    raw = uploaded_file.getvalue()
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode CSV")


def realtime_simulation_ui(results: pd.DataFrame, sleep_s: float = 0.15, max_rows: int = 200):
    logger = get_logger()
    st.subheader("Real-time detection simulation")
    st.caption("Simulates streaming input row-by-row and prints alerts. Alerts are also saved to `logs/alerts.log`.")

    if len(results) == 0:
        st.info("No rows available for simulation.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Start simulation", type="primary", disabled=st.session_state.sim_running):
            st.session_state.sim_running = True
            st.session_state.sim_idx = 0
            st.rerun()
    with c2:
        if st.button("Stop", disabled=not st.session_state.sim_running):
            st.session_state.sim_running = False
            st.rerun()

    n = min(len(results), int(max_rows))
    status_box = st.empty()
    progress = st.progress(0)
    last_rows_box = st.empty()

    if not st.session_state.sim_running:
        st.caption("Click **Start simulation** to begin streaming.")
        progress.progress(min(1.0, float(st.session_state.sim_idx) / max(1, n)))
        return

    i = int(st.session_state.sim_idx)
    if i >= n:
        st.session_state.sim_running = False
        progress.progress(1.0)
        status_box.success("Simulation finished.")
        return

    row = results.iloc[i]
    score = float(row["hybrid_score"])
    pred = int(row["final_prediction"])
    level = row["risk_level"]

    if pred == -1:
        msg = (
            f"[ALERT] t={i:05d} | score={score:.3f} | risk={level} | "
            f"src={row.get('src_ip','?')} -> dst={row.get('dst_ip','?')} | "
            f"proto={row.get('protocol','?')} | size={row.get('packet_size','?')} | "
            f"duration={row.get('duration','?')} | logins={row.get('login_attempts','?')}"
        )
        logger.info(msg)
        st.toast(f"LIVE ALERT 🚨  {msg}", icon="🚨")
        status_box.error(msg)
    else:
        status_box.info(f"[OK] t={i:05d} | score={score:.3f} | risk={level}")

    progress.progress(min(1.0, (i + 1) / max(1, n)))

    start = max(0, i - 5)
    last_rows_box.dataframe(results.iloc[start : i + 1], use_container_width=True, height=220)

    st.session_state.sim_idx = i + 1
    time.sleep(max(0.0, float(sleep_s)))
    st.rerun()


def main():
    _init_state()

    st.title("Silent Guardian: Unsupervised Cyber Attack Detection System")
    st.caption("Detect unknown cyber attacks using unsupervised machine learning (Isolation Forest + Autoencoder + Hybrid scoring).")

    with st.sidebar:
        st.header("Controls")
        auto_run = st.checkbox(
            "Auto-run when settings change",
            value=False,
            help="If enabled, detection runs automatically whenever you change sliders (can be slower in autoencoder/hybrid).",
        )

        with st.form("controls_form", border=False):
            mode = st.selectbox("Detection mode", ["hybrid", "iforest", "autoencoder"], index=0)
            contamination = st.slider("IsolationForest contamination", 0.01, 0.30, 0.05, 0.01)
            threshold = st.slider("Anomaly threshold (final score)", 0.05, 0.95, 0.60, 0.01)
            ae_epochs = st.slider("Autoencoder epochs", 5, 80, 25, 5, disabled=(mode == "iforest"))
            ae_batch = st.select_slider(
                "Autoencoder batch size",
                options=[16, 32, 64, 128, 256],
                value=64,
                disabled=(mode == "iforest"),
            )
            submitted = st.form_submit_button("Run detection", type="primary")

        st.divider()
        st.subheader("Simulation")
        sim_sleep = st.slider("Stream delay (seconds)", 0.0, 1.0, 0.15, 0.05)
        sim_max_rows = st.slider("Max rows to stream", 50, 1000, 200, 50)

    tabs = st.tabs(["Data", "Simulation", "Results"])

    with tabs[0]:
        st.subheader("Data")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            use_sample = st.button("Use sample dataset", help="Generates a realistic sample CSV if you don't have one.")
        with col_b:
            sample_seed = st.number_input("Sample seed", min_value=0, max_value=999_999, value=42, step=1)
        with col_c:
            sample_attack_ratio = st.slider("Sample attack ratio", 0.00, 0.30, 0.05, 0.01)

        st.download_button(
            "Download sample CSV",
            data=to_download_bytes(generate_sample_dataset_cached(500, attack_ratio=0.05, seed=42)),
            file_name="silent_guardian_sample.csv",
            mime="text/csv",
        )

        # Streamlit reruns the whole script on every interaction (including simulation ticks).
        # File uploader still returns the same file each time — do NOT wipe results on every rerun.
        if uploaded is not None:
            sig = (getattr(uploaded, "name", ""), getattr(uploaded, "size", None))
            if st.session_state.upload_sig != sig:
                try:
                    st.session_state.df = read_csv_robust(uploaded)
                except UnicodeDecodeError:
                    st.error(
                        "CSV encoding error (not UTF-8). "
                        "This is common for Kaggle/Windows datasets. "
                        "Try exporting/resaving as UTF-8, or upload again."
                    )
                    st.stop()
                st.session_state.df_source = ("upload", getattr(uploaded, "name", "uploaded.csv"))
                st.session_state.upload_sig = sig
                st.session_state.results = None
                st.session_state.meta = None
                st.session_state.last_cfg_key = None
                st.session_state.sim_running = False
                st.session_state.sim_idx = 0
        elif use_sample:
            st.session_state.df = generate_sample_dataset_cached(
                2000, attack_ratio=float(sample_attack_ratio), seed=int(sample_seed)
            )
            st.session_state.df_source = ("sample", int(sample_seed), 2000, float(sample_attack_ratio))
            st.session_state.upload_sig = None
            st.session_state.results = None
            st.session_state.meta = None
            st.session_state.last_cfg_key = None

        if st.session_state.df is None:
            st.info("Upload a CSV or click **Use sample dataset** to get started.")
            st.stop()

        df = st.session_state.df
        st.write("Expected columns:", ", ".join(core.REQUIRED_COLUMNS))
        st.caption(f"Source: `{st.session_state.df_source}`")
        st.dataframe(df.head(30), use_container_width=True)

        cfg = make_cfg(contamination, threshold, mode, ae_epochs, ae_batch)
        cfg_key = _cfg_key(cfg)

        should_run = submitted
        if auto_run and st.session_state.results is not None and st.session_state.last_cfg_key != cfg_key:
            should_run = True
        if auto_run and st.session_state.results is None:
            should_run = True

        if should_run:
            with st.spinner("Running detection..."):
                results, meta = run_detection_cached(df, cfg)
            st.session_state.results = results
            st.session_state.meta = meta
            st.session_state.last_cfg_key = cfg_key
            st.success(f"Done. Anomalies detected: {meta['anomaly_count']} / {meta['n_rows']}")
        else:
            st.caption("Adjust settings in the sidebar and click **Run detection** (or enable auto-run).")

    results = st.session_state.results
    meta = st.session_state.meta

    with tabs[1]:
        st.subheader("Simulation")
        if results is None or meta is None:
            st.info("No results yet. Go to the **Data** tab and click **Run detection** first.")
        else:
            realtime_simulation_ui(results, sleep_s=sim_sleep, max_rows=sim_max_rows)

    with tabs[2]:
        st.subheader("Results")
        if results is None or meta is None:
            st.info("No results yet. Go to the **Data** tab and run detection.")
        elif st.session_state.get("sim_running"):
            st.warning("Simulation is running — charts refresh is skipped on each tick for speed. Stop simulation to view full results here.")
            render_kpis(results)
        else:
            render_kpis(results)

            st.subheader("Risk score summary")
            st.write(
                {
                    "threshold": float(meta["threshold"]),
                    "avg_risk": float(results["hybrid_score"].mean()),
                    "max_risk": float(results["hybrid_score"].max()),
                    "min_risk": float(results["hybrid_score"].min()),
                    "overall_risk_level": risk_level(float(results["hybrid_score"].mean())),
                }
            )

            st.subheader("Visualizations")
            v1, v2 = st.columns(2)
            with v1:
                fig = core.plot_scatter_normal_vs_anomaly(results)
                st.pyplot(fig, clear_figure=True)
            with v2:
                fig = core.plot_scores_over_time(results)
                st.pyplot(fig, clear_figure=True)

            fig = core.plot_histograms(results)
            st.pyplot(fig, clear_figure=True)

            st.subheader("Anomaly table (dynamic filters)")
            f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
            with f1:
                min_score = st.slider("Min score", 0.0, 1.0, float(meta["threshold"]), 0.01)
            with f2:
                risk = st.multiselect("Risk level", ["Low", "Medium", "High"], default=["Medium", "High"])
            with f3:
                protos = sorted([p for p in results["protocol"].dropna().astype(str).unique().tolist()])
                proto_sel = st.multiselect(
                    "Protocol",
                    protos,
                    default=(protos[: min(6, len(protos))] if protos else []),
                )
            with f4:
                q = st.text_input("Search (src/dst ip contains)", value="")

            anomalies = results[results["final_prediction"] == -1].copy()
            anomalies = anomalies[anomalies["hybrid_score"].astype(float) >= float(min_score)]
            if risk:
                anomalies = anomalies[anomalies["risk_level"].astype(str).isin(risk)]
            if proto_sel:
                anomalies = anomalies[anomalies["protocol"].astype(str).isin(proto_sel)]
            if q.strip():
                s = q.strip()
                anomalies = anomalies[
                    anomalies["src_ip"].astype(str).str.contains(s, case=False, na=False)
                    | anomalies["dst_ip"].astype(str).str.contains(s, case=False, na=False)
                ]

            try:
                st.dataframe(
                    anomalies.style.apply(color_anomaly_rows, axis=1),
                    use_container_width=True,
                    height=380,
                )
            except AttributeError as e:
                if "requires jinja2" in str(e):
                    st.warning("Row highlighting needs `jinja2`. Install it via `pip install -r requirements.txt`.")
                    st.dataframe(anomalies, use_container_width=True, height=380)
                else:
                    raise

            cdl1, cdl2 = st.columns([1, 1])
            with cdl1:
                st.download_button(
                    "Download results CSV",
                    data=to_download_bytes(results),
                    file_name="silent_guardian_results.csv",
                    mime="text/csv",
                )
            with cdl2:
                st.download_button(
                    "Download anomalies CSV",
                    data=to_download_bytes(anomalies),
                    file_name="silent_guardian_anomalies.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()

