import ipaddress
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "packet_size",
    "duration",
    "src_ip",
    "dst_ip",
    "login_attempts",
    "protocol",
]


def ensure_logs_dir(log_dir: str = "logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_logger(name: str = "silent_guardian", log_dir: str = "logs") -> logging.Logger:
    log_dir = ensure_logs_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(os.path.join(log_dir, "alerts.log"), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger


def ip_to_int(ip: str) -> int:
    """
    Convert an IPv4/IPv6 string to a stable integer.
    Falls back to a bounded hash for invalid values.
    """
    if ip is None or (isinstance(ip, float) and np.isnan(ip)):
        return 0
    s = str(ip).strip()
    if not s:
        return 0
    try:
        return int(ipaddress.ip_address(s))
    except ValueError:
        # Keep it deterministic and bounded so scaling behaves reasonably
        return abs(hash(s)) % (2**31 - 1)


def risk_level(score_0_to_1: float) -> str:
    if score_0_to_1 < 0.35:
        return "Low"
    if score_0_to_1 < 0.70:
        return "Medium"
    return "High"


def validate_and_align_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )
    return df[REQUIRED_COLUMNS].copy()


def generate_sample_dataset(
    n_rows: int = 2000,
    attack_ratio: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic-ish network activity dataset.
    Note: no labels are required/used for training, but we include an optional
    'ground_truth' column for your own sanity checks; it's excluded by default
    during preprocessing in main/app.
    """
    rng = np.random.default_rng(seed)

    protocols = np.array(["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "SSH", "DNS"])

    def rand_ipv4() -> str:
        return ".".join(map(str, rng.integers(1, 255, size=4)))

    is_attack = rng.random(n_rows) < attack_ratio

    # Normal behavior
    packet_size = rng.normal(loc=900, scale=250, size=n_rows).clip(40, 1500)
    duration = rng.lognormal(mean=0.3, sigma=0.7, size=n_rows).clip(0.01, 60.0)
    login_attempts = rng.poisson(lam=1.2, size=n_rows).clip(0, 12)
    protocol = rng.choice(protocols, size=n_rows, p=np.array([0.28, 0.22, 0.04, 0.18, 0.14, 0.08, 0.06]))

    # Attacks: more extreme distributions (bigger packets, longer/shorter durations,
    # brute force logins, rare protocols/odd patterns)
    packet_size[is_attack] = rng.normal(loc=1350, scale=180, size=is_attack.sum()).clip(200, 1500)
    duration[is_attack] = rng.lognormal(mean=1.0, sigma=0.9, size=is_attack.sum()).clip(0.01, 120.0)
    login_attempts[is_attack] = rng.poisson(lam=6.0, size=is_attack.sum()).clip(0, 30)
    protocol[is_attack] = rng.choice(np.array(["SSH", "ICMP", "DNS", "TCP"]), size=is_attack.sum(), p=np.array([0.35, 0.25, 0.20, 0.20]))

    src_ip = np.array([rand_ipv4() for _ in range(n_rows)], dtype=object)
    dst_ip = np.array([rand_ipv4() for _ in range(n_rows)], dtype=object)

    # Add a few "hot" destinations to mimic scanning / targeted attacks
    hot_targets = np.array(["10.0.0.5", "10.0.0.10", "172.16.0.3", "192.168.1.1"], dtype=object)
    if is_attack.any():
        dst_ip[is_attack] = rng.choice(hot_targets, size=is_attack.sum())

    df = pd.DataFrame(
        {
            "packet_size": packet_size.astype(float),
            "duration": duration.astype(float),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "login_attempts": login_attempts.astype(int),
            "protocol": protocol,
            "ground_truth": is_attack.astype(int),
        }
    )

    # Introduce a tiny amount of missingness
    for col in ["packet_size", "duration", "protocol", "src_ip"]:
        idx = rng.choice(n_rows, size=max(1, n_rows // 200), replace=False)
        df.loc[idx, col] = np.nan

    return df


def to_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@dataclass(frozen=True)
class DetectionConfig:
    contamination: float = 0.05
    threshold: float = 0.60
    model_mode: str = "hybrid"  # "iforest" | "autoencoder" | "hybrid"
    ae_epochs: int = 25
    ae_batch_size: int = 64
    ae_validation_split: float = 0.1
    random_state: int = 42


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

