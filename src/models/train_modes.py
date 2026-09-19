"""
train_modes.py
==============
Trains and validates the Prevalidated Multi-Tier Detection Models
and the Research Baselines for academic comparison.

Prevalidated Modes:
  - Mode 1: Full-Feature Random Forest on F_full (78 features)
  - Mode 2: Reduced L3/L4 Random Forest on F_degraded (16 features)
  - Mode 3: Deterministic Fallback Rule Engine on F_fallback (4 signals)

Research Baselines:
  - Baseline 1: Static Full Model (Zero-fill on degradation)
  - Baseline 2: Mean Imputation + Full Model
  - Baseline 3: Dropout-Trained Model (Trained with 30% synthetic feature dropout)
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from src.data_loader import (
    generate_calibrated_benchmark_dataset,
    get_train_test_splits,
    DEGRADED_INDICES,
    FALLBACK_INDICES,
    ATTACK_CLASSES
)
from src.models.fallback_engine import FallbackRuleEngine


def train_and_save_all_models(models_dir=None, num_samples=15000):
    """
    Trains all prevalidated modes and research baselines, saving them to disk.
    """
    if models_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"[*] Generating {num_samples} calibrated training/validation samples...")
    X, y = generate_calibrated_benchmark_dataset(num_samples=num_samples, random_seed=42)
    X_train, y_train, X_val, y_val = get_train_test_splits(X, y, test_ratio=0.25)
    
    print(f"[*] Training Data Shape: {X_train.shape}, Labels: {len(y_train)}")
    
    # -------------------------------------------------------------
    # 1. Mode 1: Full-Feature Random Forest (F_full)
    # -------------------------------------------------------------
    print("[1/5] Training Mode 1 (Full-Feature ML: 78 features)...")
    m1 = RandomForestClassifier(n_estimators=50, max_depth=14, n_jobs=-1, random_state=42)
    m1.fit(X_train, y_train)
    score1 = m1.score(X_val, y_val)
    print(f"      Mode 1 Validation Accuracy (Clean): {score1:.4f}")
    joblib.dump(m1, os.path.join(models_dir, "mode1_full.pkl"))
    
    # -------------------------------------------------------------
    # 2. Mode 2: Reduced L3/L4 Random Forest (F_degraded)
    # -------------------------------------------------------------
    print("[2/5] Training Mode 2 (Reduced L3/L4 ML: 16 core features)...")
    X_train_deg = X_train[:, DEGRADED_INDICES]
    X_val_deg = X_val[:, DEGRADED_INDICES]
    m2 = RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42)
    m2.fit(X_train_deg, y_train)
    score2 = m2.score(X_val_deg, y_val)
    print(f"      Mode 2 Validation Accuracy (Clean): {score2:.4f}")
    joblib.dump(m2, os.path.join(models_dir, "mode2_degraded.pkl"))
    
    # -------------------------------------------------------------
    # 3. Mode 3: Fallback Deterministic Engine
    # -------------------------------------------------------------
    print("[3/5] Instantiating Mode 3 (Deterministic Fallback Engine)...")
    m3 = FallbackRuleEngine()
    joblib.dump(m3, os.path.join(models_dir, "mode3_fallback.pkl"))
    
    # -------------------------------------------------------------
    # 4. Baseline 2: Imputer Pipeline (Mean Imputation on Missing Features)
    # -------------------------------------------------------------
    print("[4/5] Training Baseline 2 (Mean Imputer)...")
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    joblib.dump(imputer, os.path.join(models_dir, "baseline_imputer.pkl"))
    
    # -------------------------------------------------------------
    # 5. Baseline 3: Dropout-Trained Model (Trained with 30% Random Missingness)
    # -------------------------------------------------------------
    print("[5/5] Training Baseline 3 (Feature-Dropout Model: 30% Masking)...")
    X_train_drop = X_train.copy()
    drop_mask = np.random.binomial(n=1, p=0.30, size=X_train.shape).astype(bool)
    X_train_drop[drop_mask] = 0.0
    
    m_drop = RandomForestClassifier(n_estimators=50, max_depth=14, n_jobs=-1, random_state=42)
    m_drop.fit(X_train_drop, y_train)
    score_drop = m_drop.score(X_val, y_val)
    print(f"      Baseline 3 Validation Accuracy: {score_drop:.4f}")
    joblib.dump(m_drop, os.path.join(models_dir, "baseline_dropout.pkl"))
    
    # Save validation test holdout for reproducible benchmarking
    np.savez_compressed(
        os.path.join(models_dir, "eval_test_data.npz"),
        X_val=X_val,
        y_val=y_val
    )
    print(f"[+] All models and test data successfully trained and persisted in: {models_dir}")
    return models_dir


if __name__ == "__main__":
    train_and_save_all_models()
