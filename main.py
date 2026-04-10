import argparse
import time
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from utils import (
    DetectionConfig,
    REQUIRED_COLUMNS,
    clamp01,
    generate_sample_dataset,
    get_logger,
    ip_to_int,
    risk_level,
    validate_and_align_columns,
)


def build_preprocessor() -> Tuple[ColumnTransformer, list, list]:
    numeric_cols = ["packet_size", "duration", "login_attempts", "src_ip_int", "dst_ip_int"]
    categorical_cols = ["protocol"]

    numeric_pipe = Pipeline(
        steps=[
            # scikit-learn imputers sometimes receive read-only views (esp. from pandas dtypes);
            # copy to a writable ndarray so missing-value filling works reliably.
            ("to_numpy", FunctionTransformer(lambda x: np.asarray(x).copy(), feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("to_numpy", FunctionTransformer(lambda x: np.asarray(x).copy(), feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align columns, encode IPs to integers, and keep only relevant fields.
    """
    # Drop optional column if present
    if "ground_truth" in df.columns:
        df = df.drop(columns=["ground_truth"])

    df = validate_and_align_columns(df)

    # Keep as plain object dtype and ensure missing values are numpy NaN.
    # Avoid pandas 'string' dtype here because it introduces pd.NA, which can
    # break scikit-learn missing-value masking in some versions.
    for col in ["src_ip", "dst_ip", "protocol"]:
        df[col] = df[col].astype(object)
        df[col] = df[col].where(pd.notna(df[col]), np.nan)

    # Ensure numeric columns are numeric (bad values -> NaN -> imputed)
    for col in ["packet_size", "duration", "login_attempts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["src_ip_int"] = df["src_ip"].map(ip_to_int).astype("int64")
    df["dst_ip_int"] = df["dst_ip"].map(ip_to_int).astype("int64")

    return df


def fit_isolation_forest(
    X: np.ndarray,
    contamination: float,
    random_state: int,
) -> IsolationForest:
    clf = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X)
    return clf


def iforest_scores_and_preds(
    clf: IsolationForest, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    preds = clf.predict(X)  # 1 normal, -1 anomaly
    # score_samples: higher = more normal; invert to get anomaly score
    raw = -clf.score_samples(X)
    return raw, preds


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def try_import_keras():
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # noqa: F401

        return True
    except Exception:
        return False


def build_autoencoder(input_dim: int):
    from tensorflow import keras

    inp = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Dense(64, activation="relu")(inp)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    bottleneck = keras.layers.Dense(16, activation="relu")(x)

    x = keras.layers.Dense(32, activation="relu")(bottleneck)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def fit_autoencoder(
    X: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float,
    random_state: int,
):
    from tensorflow import keras
    import tensorflow as tf

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    model = build_autoencoder(X.shape[1])
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    model.fit(
        X,
        X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        verbose=0,
        callbacks=cb,
    )
    return model


def autoencoder_scores(model, X: np.ndarray) -> np.ndarray:
    recon = model.predict(X, verbose=0)
    err = np.mean(np.square(X - recon), axis=1)
    return err


def compute_hybrid_score(if_score01: np.ndarray, ae_score01: Optional[np.ndarray]) -> np.ndarray:
    if ae_score01 is None:
        return if_score01
    return 0.5 * if_score01 + 0.5 * ae_score01


def detect_anomalies(
    df_raw: pd.DataFrame,
    cfg: DetectionConfig,
) -> Tuple[pd.DataFrame, Dict]:
    logger = get_logger()
    df = prepare_dataframe(df_raw)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor()
    X = preprocessor.fit_transform(df)

    meta: Dict = {
        "used_columns": REQUIRED_COLUMNS,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_rows": int(len(df)),
        "mode": cfg.model_mode,
    }

    # Isolation Forest
    if_clf = fit_isolation_forest(X, contamination=cfg.contamination, random_state=cfg.random_state)
    if_raw, if_pred = iforest_scores_and_preds(if_clf, X)
    if_score01 = minmax01(if_raw)

    # Autoencoder (optional dependency, but implemented)
    ae_model = None
    ae_score01 = None
    if cfg.model_mode in ("autoencoder", "hybrid"):
        if not try_import_keras():
            logger.warning("TensorFlow/Keras not available; falling back to Isolation Forest only.")
        else:
            ae_model = fit_autoencoder(
                X,
                epochs=cfg.ae_epochs,
                batch_size=cfg.ae_batch_size,
                validation_split=cfg.ae_validation_split,
                random_state=cfg.random_state,
            )
            ae_raw = autoencoder_scores(ae_model, X)
            ae_score01 = minmax01(ae_raw)

    # Final scoring
    if cfg.model_mode == "iforest":
        final_score01 = if_score01
    elif cfg.model_mode == "autoencoder":
        if ae_score01 is None:
            final_score01 = if_score01
        else:
            final_score01 = ae_score01
    else:
        final_score01 = compute_hybrid_score(if_score01, ae_score01)

    thr = clamp01(cfg.threshold)
    final_pred = np.where(final_score01 >= thr, -1, 1)

    out = df_raw.copy()
    out["iforest_prediction"] = if_pred
    out["iforest_score"] = if_score01
    if ae_score01 is not None:
        out["autoencoder_score"] = ae_score01
    else:
        out["autoencoder_score"] = np.nan
    out["hybrid_score"] = final_score01
    out["final_prediction"] = final_pred
    out["risk_level"] = [risk_level(float(s)) for s in final_score01]

    meta.update(
        {
            "threshold": thr,
            "contamination": cfg.contamination,
            "anomaly_count": int((final_pred == -1).sum()),
        }
    )
    return out, meta


def plot_scatter_normal_vs_anomaly(df: pd.DataFrame, score_col: str = "hybrid_score"):
    # Use two intuitive axes: packet_size vs duration
    normal = df[df["final_prediction"] == 1]
    anom = df[df["final_prediction"] == -1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(normal["packet_size"], normal["duration"], s=18, alpha=0.6, label="Normal", marker="o")
    ax.scatter(anom["packet_size"], anom["duration"], s=45, alpha=0.9, label="Anomaly", marker="x")
    ax.set_title("Normal vs Anomaly (packet_size vs duration)")
    ax.set_xlabel("packet_size")
    ax.set_ylabel("duration")
    ax.legend()
    return fig


def plot_histograms(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    cols = ["packet_size", "duration", "login_attempts", "protocol"]
    for ax, col in zip(axes, cols):
        if col == "protocol":
            df[col].astype(str).value_counts().head(10).plot(kind="bar", ax=ax)
            ax.set_title("protocol (top 10)")
            ax.tick_params(axis="x", rotation=30)
        else:
            ax.hist(df[col].astype(float), bins=30, alpha=0.8, color="#4C78A8")
            ax.set_title(col)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Feature Distributions", y=1.02)
    fig.tight_layout()
    return fig


def plot_scores_over_time(df: pd.DataFrame, score_col: str = "hybrid_score"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df[score_col].values, linewidth=1.2)
    ax.set_title(f"Anomaly Scores Over Time ({score_col})")
    ax.set_xlabel("time (row index)")
    ax.set_ylabel("score (0-1)")
    ax.grid(True, alpha=0.3)
    return fig


def simulate_realtime_stream(
    df_results: pd.DataFrame,
    sleep_s: float = 0.25,
    max_rows: Optional[int] = 200,
):
    logger = get_logger()
    n = len(df_results) if max_rows is None else min(len(df_results), max_rows)
    print("\n--- Real-Time Detection Simulation (console) ---")
    print("Press CTRL+C to stop.\n")
    try:
        for i in range(n):
            row = df_results.iloc[i]
            score = float(row["hybrid_score"])
            pred = int(row["final_prediction"])
            level = row["risk_level"]
            if pred == -1:
                msg = (
                    f"[ALERT] t={i:05d} | score={score:.3f} | risk={level:<6} | "
                    f"src={row.get('src_ip','?')} -> dst={row.get('dst_ip','?')} | "
                    f"proto={row.get('protocol','?')} | size={row.get('packet_size','?')} | "
                    f"duration={row.get('duration','?')} | logins={row.get('login_attempts','?')}"
                )
                print(msg)
                logger.info(msg)
            else:
                print(f"[OK]    t={i:05d} | score={score:.3f} | risk={level:<6}")
            time.sleep(max(0.0, float(sleep_s)))
    except KeyboardInterrupt:
        print("\nSimulation stopped.")


def load_csv_or_generate(path: Optional[str], n_rows: int = 2000) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        return df
    return generate_sample_dataset(n_rows=n_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Silent Guardian: Unsupervised Cyber Attack Detection")
    p.add_argument("--csv", type=str, default=None, help="Path to input CSV (optional).")
    p.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination (0-0.5).")
    p.add_argument("--threshold", type=float, default=0.60, help="Final anomaly threshold (0-1).")
    p.add_argument("--mode", type=str, default="hybrid", choices=["iforest", "autoencoder", "hybrid"])
    p.add_argument("--ae-epochs", type=int, default=25)
    p.add_argument("--ae-batch-size", type=int, default=64)
    p.add_argument("--simulate", action="store_true", help="Run a real-time simulation loop.")
    p.add_argument("--sleep", type=float, default=0.15, help="Sleep seconds between streamed rows.")
    p.add_argument("--max-rows", type=int, default=200, help="Max rows to stream in simulation.")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    p.add_argument("--out", type=str, default="results.csv", help="Output CSV filename.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = DetectionConfig(
        contamination=float(args.contamination),
        threshold=float(args.threshold),
        model_mode=str(args.mode),
        ae_epochs=int(args.ae_epochs),
        ae_batch_size=int(args.ae_batch_size),
    )

    df = load_csv_or_generate(args.csv)
    results, meta = detect_anomalies(df, cfg)

    print("\n=== Silent Guardian Report ===")
    print(f"Rows: {meta['n_rows']}")
    print(f"Mode: {meta['mode']}")
    print(f"Contamination: {meta['contamination']}")
    print(f"Threshold: {meta['threshold']}")
    print(f"Anomalies detected: {meta['anomaly_count']}")
    print("Saved alerts log to: logs/alerts.log")

    results.to_csv(args.out, index=False)
    print(f"Saved results to: {args.out}")

    if not args.no_plots:
        fig1 = plot_scatter_normal_vs_anomaly(results)
        fig2 = plot_histograms(results)
        fig3 = plot_scores_over_time(results)
        plt.show()

    if args.simulate:
        simulate_realtime_stream(results, sleep_s=args.sleep, max_rows=args.max_rows)


if __name__ == "__main__":
    main()

