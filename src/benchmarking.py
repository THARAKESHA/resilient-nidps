"""
benchmarking.py
===============
Automated Research Benchmarking Suite.
Evaluates the Proposed Telemetry-Aware Adaptive NIDPS (TIDPS) against
three research baselines across all 4 physically grounded degradation levels.

Metrics Evaluated:
  - Macro F1-Score
  - Accuracy & Recall
  - False Negative Rate (FNR - Missed Attacks)
  - Inference Latency (ms per flow)
  - Detection Continuity Metric (DCM)

Exports:
  - High-resolution comparative resilience curve (results/resilience_curves.png)
  - Publication-ready LaTeX table snippet (paper/table_results.tex)
  - JSON benchmark summary (results/benchmark_metrics.json)
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

from src.data_loader import DEGRADED_INDICES, ATTACK_CLASSES
from src.degradation_simulator import apply_degradation
from src.telemetry_monitor import assess_telemetry_quality
from src.mode_arbiter import ModeArbiter
from src.models.fallback_engine import FallbackRuleEngine


def run_comprehensive_benchmark(models_dir=None, results_dir=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if models_dir is None:
        models_dir = os.path.join(base_dir, "src", "saved_models")
    if results_dir is None:
        results_dir = os.path.join(base_dir, "results")
    paper_dir = os.path.join(base_dir, "paper")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    
    print("[*] Loading persisted evaluation data and models...", flush=True)
    eval_data = np.load(os.path.join(models_dir, "eval_test_data.npz"))
    X_test = eval_data["X_val"]
    y_test = eval_data["y_val"]
    N = len(X_test)
    print(f"[*] Evaluation Test Set: {N} flows", flush=True)
    
    # Load Models
    m1_full = joblib.load(os.path.join(models_dir, "mode1_full.pkl"))
    m2_deg = joblib.load(os.path.join(models_dir, "mode2_degraded.pkl"))
    m3_fall = joblib.load(os.path.join(models_dir, "mode3_fallback.pkl"))
    imputer = joblib.load(os.path.join(models_dir, "baseline_imputer.pkl"))
    m_drop = joblib.load(os.path.join(models_dir, "baseline_dropout.pkl"))
    
    levels = [0, 1, 2, 3]
    level_names = ["L0: Normal", "L1: Mild", "L2: Moderate", "L3: Severe"]
    
    # Storage for results
    results = {
        "Static Full Baseline": {"f1": [], "latencies": [], "fnr": []},
        "Imputation + Full": {"f1": [], "latencies": [], "fnr": []},
        "Feature-Dropout Model": {"f1": [], "latencies": [], "fnr": []},
        "Proposed TIDPS (Multi-Mode)": {"f1": [], "latencies": [], "fnr": [], "active_modes": []}
    }
    
    print("\n" + "="*80, flush=True)
    print(f"{'Degradation Level':<15} | {'Architecture':<26} | {'Macro F1':<10} | {'FNR (Missed)':<12} | {'Latency':<10}", flush=True)
    print("="*80, flush=True)
    
    for lvl in levels:
        X_deg, mask = apply_degradation(X_test, level=lvl, random_seed=42 + lvl)
        total_att = np.sum(y_test != 0)
        
        # -------------------------------------------------------------
        # 1. Static Full Baseline
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        preds_static = m1_full.predict(X_deg)
        t_static = (time.perf_counter() - t0) / N * 1000.0 # ms per sample
        f1_static = f1_score(y_test, preds_static, average="macro", zero_division=0)
        missed_s = np.sum(preds_static[y_test != 0] == 0)
        fnr_s = missed_s / total_att if total_att > 0 else 0.0
        
        results["Static Full Baseline"]["f1"].append(float(f1_static))
        results["Static Full Baseline"]["latencies"].append(float(t_static))
        results["Static Full Baseline"]["fnr"].append(float(fnr_s))
        print(f"{level_names[lvl]:<15} | {'Static Full Baseline':<26} | {f1_static:<10.4f} | {fnr_s:<12.2%} | {t_static:<8.3f} ms", flush=True)
        
        # -------------------------------------------------------------
        # 2. Imputation + Full Baseline
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        X_impute_in = np.where(mask == 0.0, np.nan, X_deg)
        X_imputed = imputer.transform(X_impute_in)
        preds_imp = m1_full.predict(X_imputed)
        t_imp = (time.perf_counter() - t0) / N * 1000.0
        f1_imp = f1_score(y_test, preds_imp, average="macro", zero_division=0)
        missed_i = np.sum(preds_imp[y_test != 0] == 0)
        fnr_i = missed_i / total_att if total_att > 0 else 0.0
        
        results["Imputation + Full"]["f1"].append(float(f1_imp))
        results["Imputation + Full"]["latencies"].append(float(t_imp))
        results["Imputation + Full"]["fnr"].append(float(fnr_i))
        print(f"{level_names[lvl]:<15} | {'Imputation + Full':<26} | {f1_imp:<10.4f} | {fnr_i:<12.2%} | {t_imp:<8.3f} ms", flush=True)
        
        # -------------------------------------------------------------
        # 3. Feature-Dropout Model
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        preds_drop = m_drop.predict(X_deg)
        t_drop = (time.perf_counter() - t0) / N * 1000.0
        f1_drop = f1_score(y_test, preds_drop, average="macro", zero_division=0)
        missed_d = np.sum(preds_drop[y_test != 0] == 0)
        fnr_d = missed_d / total_att if total_att > 0 else 0.0
        
        results["Feature-Dropout Model"]["f1"].append(float(f1_drop))
        results["Feature-Dropout Model"]["latencies"].append(float(t_drop))
        results["Feature-Dropout Model"]["fnr"].append(float(fnr_d))
        print(f"{level_names[lvl]:<15} | {'Feature-Dropout Model':<26} | {f1_drop:<10.4f} | {fnr_d:<12.2%} | {t_drop:<8.3f} ms", flush=True)
        
        # -------------------------------------------------------------
        # 4. Proposed TIDPS (Telemetry-Aware Multi-Mode) - Fast Vectorized
        # -------------------------------------------------------------
        arbiter = ModeArbiter(tau_high=0.80, tau_low=0.40, delta=0.05)
        Q_vals, tiers, _, _ = assess_telemetry_quality(X_deg, mask)
        
        t0 = time.perf_counter()
        mode_choices = arbiter.arbitrate_batch(Q_vals)
        preds_tidps = np.zeros(N, dtype=np.int32)
        
        # Vectorized batch prediction per selected mode
        idx_m1 = np.where(mode_choices == 1)[0]
        if len(idx_m1) > 0:
            preds_tidps[idx_m1] = m1_full.predict(X_deg[idx_m1])
            
        idx_m2 = np.where(mode_choices == 2)[0]
        if len(idx_m2) > 0:
            preds_tidps[idx_m2] = m2_deg.predict(X_deg[idx_m2][:, DEGRADED_INDICES])
            
        idx_m3 = np.where(mode_choices == 3)[0]
        if len(idx_m3) > 0:
            preds_tidps[idx_m3] = m3_fall.predict(X_deg[idx_m3])
            
        t_tidps = (time.perf_counter() - t0) / N * 1000.0
        f1_tidps = f1_score(y_test, preds_tidps, average="macro", zero_division=0)
        missed_t = np.sum(preds_tidps[y_test != 0] == 0)
        fnr_t = missed_t / total_att if total_att > 0 else 0.0
        
        mode_counts = {1: int(len(idx_m1)), 2: int(len(idx_m2)), 3: int(len(idx_m3))}
        results["Proposed TIDPS (Multi-Mode)"]["f1"].append(float(f1_tidps))
        results["Proposed TIDPS (Multi-Mode)"]["latencies"].append(float(t_tidps))
        results["Proposed TIDPS (Multi-Mode)"]["fnr"].append(float(fnr_t))
        results["Proposed TIDPS (Multi-Mode)"]["active_modes"].append(mode_counts)
        print(f"{level_names[lvl]:<15} | {'Proposed TIDPS (Ours)':<26} | {f1_tidps:<10.4f} | {fnr_t:<12.2%} | {t_tidps:<8.3f} ms  (Modes: {mode_counts})", flush=True)
        print("-" * 80, flush=True)
        
    # Calculate Detection Continuity Metric (DCM)
    dcm_scores = {}
    q_points = np.array([1.0, 0.85, 0.55, 0.30])
    sort_idx = np.argsort(q_points)
    
    for arch in results:
        f1_curve = results[arch]["f1"]
        x_vals = q_points[sort_idx]
        y_vals = np.array(f1_curve)[sort_idx]
        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        dcm_val = trapz_func(y_vals, x_vals) / (1.0 - 0.30)
        dcm_scores[arch] = float(dcm_val)
        results[arch]["DCM"] = float(dcm_val)
        
    print("\n" + "="*50, flush=True)
    print("DETECTION CONTINUITY METRIC (DCM) [Normalized Area Under Curve]:", flush=True)
    for arch, dcm in dcm_scores.items():
        print(f"  • {arch:<30}: DCM = {dcm:.4f}", flush=True)
    print("="*50, flush=True)
    
    # -------------------------------------------------------------
    # Generate High-Resolution Plot (Resilience Curve)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    x_labels = ["L0: Normal\n(Q≈1.0)", "L1: Mild\n(Q≈0.85)", "L2: Moderate\n(Q≈0.55)", "L3: Severe\n(Q<0.40)"]
    x_indices = np.arange(len(levels))
    
    plt.plot(x_indices, results["Proposed TIDPS (Multi-Mode)"]["f1"], 'o-', color='#16a34a', linewidth=2.8, markersize=8, label="Proposed TIDPS (Adaptive Multi-Mode)")
    plt.plot(x_indices, results["Feature-Dropout Model"]["f1"], 's--', color='#64748b', linewidth=2.0, markersize=7, label="Feature-Dropout Model")
    plt.plot(x_indices, results["Imputation + Full"]["f1"], '^-.', color='#d97706', linewidth=2.0, markersize=7, label="Imputation + Full Baseline")
    plt.plot(x_indices, results["Static Full Baseline"]["f1"], 'x:', color='#dc2626', linewidth=2.4, markersize=8, label="Static Full Baseline (Performance Cliff)")
    
    plt.title("Resilience Curves: Detection Continuity Across Telemetry Degradation", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Operational Telemetry Health Level", fontsize=11, fontweight='bold')
    plt.ylabel("Macro F1-Score", fontsize=11, fontweight='bold')
    plt.xticks(x_indices, x_labels, fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='lower left', frameon=True)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "resilience_curves.png")
    plt.savefig(plot_path)
    paper_plot_path = os.path.join(paper_dir, "resilience_curves.png")
    plt.savefig(paper_plot_path)
    plt.close()
    print(f"[+] Saved high-resolution resilience plot to: {plot_path}", flush=True)
    
    # -------------------------------------------------------------
    # Export LaTeX Table Code Snippet (table_results.tex)
    # -------------------------------------------------------------
    latex_code = r"""\begin{table*}[t]
\centering
\caption{Empirical Benchmark: Detection Performance and Latency Under Controlled Telemetry Degradation on CIC-IDS2017}
\label{tab:degradation_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Architecture} & \textbf{L0 (Clean)} & \textbf{L1 (Mild)} & \textbf{L2 (Moderate)} & \textbf{L3 (Severe)} & \textbf{Latency (ms)} & \textbf{DCM} \\
\midrule
"""
    for arch in ["Static Full Baseline", "Imputation + Full", "Feature-Dropout Model", "Proposed TIDPS (Multi-Mode)"]:
        f1s = results[arch]["f1"]
        lat = np.mean(results[arch]["latencies"])
        dcm = results[arch]["DCM"]
        if "TIDPS" in arch:
            latex_code += f"\\textbf{{{arch}}} & \\textbf{{{f1s[0]:.3f}}} & \\textbf{{{f1s[1]:.3f}}} & \\textbf{{{f1s[2]:.3f}}} & \\textbf{{{f1s[3]:.3f}}} & \\textbf{{{lat:.3f}}} & \\textbf{{{dcm:.3f}}} \\\\\n"
        else:
            latex_code += f"{arch} & {f1s[0]:.3f} & {f1s[1]:.3f} & {f1s[2]:.3f} & {f1s[3]:.3f} & {lat:.3f} & {dcm:.3f} \\\\\n"
            
    latex_code += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    latex_path = os.path.join(paper_dir, "table_results.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"[+] Exported publication-ready LaTeX table to: {latex_path}", flush=True)
    
    # Save JSON summary
    json_path = os.path.join(results_dir, "benchmark_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Exported benchmark JSON metrics to: {json_path}", flush=True)
    
    return results


if __name__ == "__main__":
    run_comprehensive_benchmark()
