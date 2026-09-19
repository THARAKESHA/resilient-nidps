# 🛡️ Resilient Telemetry-Aware NIDPS (TIDPS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.64.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: CIC-IDS2017](https://img.shields.io/badge/Dataset-CIC--IDS2017-orange.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)

**Resilient AI-Based Network Intrusion Detection & Prevention System with Telemetry-Aware Multi-Mode Arbitration and Intelligent Alerting.**

---

## 📌 Abstract
Contemporary Machine Learning-based Network Intrusion Detection Systems (ML-NIDS) are conventionally trained and evaluated under the implicit assumption that network telemetry is continuously available, complete, and uncorrupted. However, in real-world production networks, line-rate packet sampling (sFlow 1-in-N), sensor queue saturation, pervasive payload encryption (TLS 1.3), and asymmetric BGP routing routinely degrade telemetry. 

Under such stress, traditional static ML models suffer catastrophic **performance cliffs** (collapsing from 99% down to 36% F1). Existing mitigation techniques—namely real-time data imputation (KNN/MICE)—incur intolerable 28ms latency bottlenecks and miss up to 86% of attacks.

**TIDPS** solves this by introducing a **telemetry-quality-driven multi-mode arbitration architecture** that maintains **detection continuity** without the line-rate latency penalty of imputation:
1. **Telemetry Quality Monitor (TQM):** Evaluates real-time flow health into an objective score $Q(t) \in [0, 1]$.
2. **Dynamic Arbiter with Hysteresis:** Eliminates high-frequency mode flapping with a $\pm 0.05$ stability band.
3. **Tri-Mode Prevalidated Detection:**
   - **Mode 1 (Full):** Tuned Random Forest on full 78 flow features ($Q \ge 0.80$).
   - **Mode 2 (Degraded):** Reduced Random Forest on 16 immutable L3/L4 headers ($0.40 \le Q < 0.80$).
   - **Mode 3 (Fallback):** Deterministic volumetric threshold rules executing in $< 5\,\mu\text{s}$ ($Q < 0.40$).
4. **Adaptive Dual-Action Mitigation:** High-severity IP Quarantine (firewall `DROP`) vs. Medium-severity Token-Bucket Rate-Limiting ($5\text{ pkts/s}$ to protect shared NAT/Wi-Fi).

---

## 🏗️ Architectural Pipeline

```mermaid
flowchart LR
    Traffic[Network Flow] --> TQM[Telemetry Quality Monitor\nComputes Q(t)]
    TQM --> Arbiter[Dynamic Arbiter\n±0.05 Hysteresis]
    Arbiter -->|Q >= 0.80| M1[Mode 1: Full ML\n78 Features]
    Arbiter -->|0.40 <= Q < 0.80| M2[Mode 2: Degraded ML\n16 L3/L4 Features]
    Arbiter -->|Q < 0.40| M3[Mode 3: Fallback Rules\nVolumetric Invariants]
    M1 & M2 & M3 --> Prev[Adaptive Prevention Engine]
    Prev --> Drop[Firewall DROP IP]
    Prev --> Rate[Token-Bucket Rate Limit]
```

---

## 📊 Empirical Benchmark Results (CIC-IDS2017)

Evaluated across 3,750 held-out flows under controlled operational degradation:

| Architecture | L0 (Clean) | L1 (Mild) | L2 (Asymmetric) | L3 (Starvation) | Latency | DCM Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Full Baseline** | 1.000 | 0.995 | 0.747 | 0.366 *(CLIFF)* | 0.008 ms | 0.7859 |
| **Imputation + Full** | 1.000 | 0.988 | 0.680 *(44% Missed)* | 0.317 *(86% Missed)* | 28.400 ms | 0.7485 |
| **Feature-Dropout Model** | 1.000 | 0.998 | 0.999 | 0.849 | 0.007 ms | 0.9723 |
| **Proposed TIDPS (Ours)** | **1.000** | **0.995** | **0.915** | **0.616** | **0.008 ms** | **0.8965** |

* **DCM (Detection Continuity Metric):** Area under the F1 curve normalized by ideal retention:
  $$\text{DCM} = \frac{\int_{0}^{1} F_1(Q) \, dQ}{\int_{0}^{1} F_1^{\text{ideal}}(Q) \, dQ}$$

---

## 🚀 Quickstart & Usage

### 1. Local Run
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/resilient-nidps.git
cd resilient-nidps

# Install dependencies
pip install -r requirements.txt

# Launch interactive dashboard
streamlit run app.py
```
* **Windows One-Click:** Double-click `run_app.bat`
* **Linux One-Click:** Execute `./run_app.sh`

### 2. Reproduce Benchmark Experiments
```bash
python -m src.benchmarking
```
Outputs high-res vector plot to `results/resilience_curves.png` and LaTeX table to `paper/table_results.tex`.

---

## 📁 Repository Structure
```text
├── app.py                      # Interactive Streamlit security dashboard
├── requirements.txt            # Python package dependencies
├── run_app.bat                 # One-click Windows launcher
├── run_app.sh                  # One-click Linux launcher
├── src/
│   ├── data_loader.py          # CIC-IDS2017 feature partitioning & dataset generator
│   ├── degradation_simulator.py# L0-L3 physical network failure models
│   ├── telemetry_monitor.py    # TQM: Completeness C(t), Consistency S(t), Score Q(t)
│   ├── mode_arbiter.py         # Dynamic mode arbiter with hysteresis buffer
│   ├── prevention_alerting.py  # IP quarantine & token-bucket rate limiting
│   ├── benchmarking.py         # Comprehensive research benchmark suite
│   └── models/
│       ├── fallback_engine.py  # Mode 3 deterministic rule engine (<5μs)
│       ├── train_modes.py      # Training script for mode zoo & baselines
│       └── saved_models/       # Persisted pre-trained model checkpoints
├── results/                    # Generated benchmark curves and metrics JSON
└── paper/                      # IEEE LaTeX manuscript, references.bib, and PDF
```

---

## 📜 Citation
```bibtex
@article{tidps2026resilient,
  title={Resilient Network Intrusion Detection via Telemetry-Aware Multi-Mode Arbitration and Graceful Degradation},
  author={Anonymous},
  journal={Department of Computer Science and Engineering},
  year={2026}
}
```
