# Silent Guardian: Unsupervised Cyber Attack Detection System

Silent Guardian is a **production-ready, resume/demo-friendly** AIML project that detects **unknown cyber attacks** using **unsupervised anomaly detection**.

It supports:
- **CSV upload input**
- **Automatic sample dataset generation**
- Two unsupervised models:
  - **Isolation Forest** (mandatory)
  - **Autoencoder (TensorFlow/Keras)** (implemented)
- **Hybrid mode** (average anomaly scores + threshold)
- **Real-time streaming simulation**
- **Console + file logging** for detected attacks
- **Streamlit dashboard** with visualizations and downloads

---

## Project Structure

- `main.py` — preprocessing + ML models + visualizations + real-time simulation
- `app.py` — Streamlit dashboard (upload, run detection, charts, anomaly table, downloads)
- `utils.py` — helpers (dataset generator, IP encoding, risk levels, logging)
- `requirements.txt` — dependencies
- `logs/alerts.log` — generated at runtime (detected attack logs)

---

## Dataset Format

Your CSV must contain these columns:

| Column | Type | Notes |
|---|---|---|
| `packet_size` | number | bytes |
| `duration` | number | seconds |
| `src_ip` | string | IPv4/IPv6 (encoded to integer) |
| `dst_ip` | string | IPv4/IPv6 (encoded to integer) |
| `login_attempts` | number | attempts count |
| `protocol` | string | e.g., TCP/UDP/HTTP/SSH |

Optional:
- `ground_truth` (0/1) can exist in the file; it will be ignored for training/inference.

---

## How It Works (Pipeline)

### Preprocessing
- **Missing values**: numeric → median, protocol → most frequent
- **Encoding**:
  - `src_ip`, `dst_ip` → stable integer conversion (IPv4/IPv6 supported)
  - `protocol` → one-hot encoding
- **Scaling**: `StandardScaler` for numeric features
- **Feature selection**: uses only the required columns

### Models

#### 1) Isolation Forest
- Produces:
  - `iforest_prediction` (1 normal, -1 anomaly)
  - `iforest_score` (normalized 0-1 anomaly score)

#### 2) Autoencoder (TensorFlow/Keras)
- Trained to reconstruct “normal-like” behavior
- Uses **reconstruction error** as anomaly score (`autoencoder_score`)

#### Hybrid Mode
- Final score = average of normalized scores:
  - `hybrid_score = 0.5 * iforest_score + 0.5 * autoencoder_score`
- Final decision:
  - `final_prediction = -1` if `hybrid_score >= threshold`

### Risk Levels
- Low / Medium / High based on the final score.

---

## Run Instructions

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run CLI (matplotlib + optional simulation)

Run with a generated dataset:

```bash
python main.py
```

Run with your CSV:

```bash
python main.py --csv your_data.csv
```

Useful options:

```bash
python main.py --csv your_data.csv --mode hybrid --contamination 0.05 --threshold 0.60
python main.py --simulate --sleep 0.15 --max-rows 200
python main.py --no-plots
```

Outputs:
- `results.csv` (default) with scores/predictions
- `logs/alerts.log` with detected attack alerts

### 3) Run Streamlit Dashboard

```bash
streamlit run app.py
```

Dashboard includes:
- Upload CSV
- Preview dataset
- Threshold slider + contamination slider
- Run detection button
- Charts (scatter, histograms, score over time)
- Anomaly table (highlighted red)
- Download results button
- Real-time simulation button

---

## Notes
- This is **unsupervised**: it assumes most traffic is normal and flags deviations.
- For best results on real datasets:
  - ensure numeric units are consistent
  - keep contamination small (e.g., 0.01–0.10)
  - tune threshold based on acceptable alert volume

