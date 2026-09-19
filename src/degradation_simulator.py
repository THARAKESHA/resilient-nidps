"""
degradation_simulator.py
========================
Physically grounded network degradation simulator.
Simulates real-world network operational failure modes:
  - Level 0 (Normal): Clean, uncorrupted telemetry (Q ≈ 1.0)
  - Level 1 (Mild): Buffer queue delay, jitter, and timestamp noise (Q ≈ 0.85)
  - Level 2 (Moderate): Asymmetric routing (backward flow loss) + 1-in-N sampling (Q ≈ 0.55)
  - Level 3 (Severe): Sensor CPU starvation, memory exhaustion, and DPI stripping (Q < 0.40)
"""

import numpy as np
from src.data_loader import (
    BACKWARD_FEATURE_INDICES,
    TIMING_FEATURE_INDICES,
    PAYLOAD_DPI_INDICES
)


def apply_degradation(X, level=0, random_seed=42):
    """
    Applies a structured, physically grounded degradation level to telemetry matrix X.
    Returns:
      X_degraded: The degraded feature matrix.
      mask: Binary observability mask (1 = observed, 0 = missing/dropped).
    """
    np.random.seed(random_seed)
    N, D = X.shape
    X_deg = X.copy()
    mask = np.ones((N, D), dtype=np.float32)
    
    if level == 0:
        # Level 0: Normal / Pristine (Q ≈ 1.0)
        # Clean telemetry, minimal measurement jitter
        return X_deg, mask
        
    elif level == 1:
        # Level 1: Mild Degradation (Q ≈ 0.85)
        # Network buffer queuing and clock drift
        for idx in TIMING_FEATURE_INDICES:
            # Jitter noise added to IAT and timing stats
            noise = np.random.normal(loc=0.0, scale=0.15 * (np.std(X_deg[:, idx]) + 1e-3), size=N)
            X_deg[:, idx] = np.maximum(0.0, X_deg[:, idx] + noise)
            
        # Random 10% dropout of secondary features
        rand_drop = np.random.binomial(n=1, p=0.10, size=(N, D)).astype(bool)
        # Protect port and duration
        rand_drop[:, 0] = False
        rand_drop[:, 1] = False
        X_deg[rand_drop] = 0.0
        mask[rand_drop] = 0.0
        
    elif level == 2:
        # Level 2: Moderate Degradation (Q ≈ 0.55)
        # Asymmetric routing (return packets take different path) + 1-in-N sampling
        
        # 1. Complete loss of all backward flow attributes
        for idx in BACKWARD_FEATURE_INDICES:
            X_deg[:, idx] = 0.0
            mask[:, idx] = 0.0
            
        # 2. Timing and IAT stats severely distorted by packet sampling (sFlow 1-in-N)
        for idx in TIMING_FEATURE_INDICES:
            # 50% of timing values lost, remainder heavily noisy
            drop_idx = np.random.binomial(n=1, p=0.50, size=N).astype(bool)
            X_deg[drop_idx, idx] = 0.0
            mask[drop_idx, idx] = 0.0
            noise = np.random.normal(loc=0.0, scale=0.60 * (np.std(X_deg[:, idx]) + 1e-3), size=N)
            X_deg[~drop_idx, idx] = np.maximum(0.0, X_deg[~drop_idx, idx] + noise[~drop_idx])
            
        # 3. Additional 15% random drop of other secondary features
        rand_drop = np.random.binomial(n=1, p=0.15, size=(N, D)).astype(bool)
        rand_drop[:, 0:3] = False # keep basic L3 headers
        X_deg[rand_drop] = 0.0
        mask[rand_drop] = 0.0
        
    elif level == 3:
        # Level 3: Severe Degradation (Q < 0.40)
        # Sensor CPU starvation, memory pool exhaustion, TLS 1.3 payload stripping
        
        # 1. All backward features lost
        for idx in BACKWARD_FEATURE_INDICES:
            X_deg[:, idx] = 0.0
            mask[:, idx] = 0.0
            
        # 2. All timing / IAT features lost (sensor CPU halted high-res timer)
        for idx in TIMING_FEATURE_INDICES:
            X_deg[:, idx] = 0.0
            mask[:, idx] = 0.0
            
        # 3. All payload / DPI / bulk / window features stripped
        for idx in PAYLOAD_DPI_INDICES:
            X_deg[:, idx] = 0.0
            mask[:, idx] = 0.0
            
        # 4. Severe general feature masking (65% overall loss)
        rand_drop = np.random.binomial(n=1, p=0.65, size=(N, D)).astype(bool)
        # Only raw packet rate and byte rate hardware registers survive
        rand_drop[:, 14] = False # Flow Bytes/s
        rand_drop[:, 15] = False # Flow Packets/s
        rand_drop[:, 44] = False # SYN Flag
        X_deg[rand_drop] = 0.0
        mask[rand_drop] = 0.0
        
    return X_deg, mask


def inject_custom_degradation(X, missing_rate=0.0, noise_level=0.0, drop_backward=False, drop_timing=False):
    """
    Fine-grained custom degradation injection for the Streamlit dashboard sliders.
    """
    N, D = X.shape
    X_deg = X.copy()
    mask = np.ones((N, D), dtype=np.float32)
    
    if drop_backward:
        for idx in BACKWARD_FEATURE_INDICES:
            X_deg[:, idx] = 0.0
            mask[:, idx] = 0.0
            
    if drop_timing:
        for idx in TIMING_FEATURE_INDICES:
            X_deg[:, idx] = 0.0
            mask[:, idx] = 0.0
            
    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level, size=(N, D))
        X_deg = np.maximum(0.0, X_deg + noise * np.std(X, axis=0, keepdims=True))
        
    if missing_rate > 0.0:
        drop = np.random.binomial(1, missing_rate, size=(N, D)).astype(bool)
        drop[:, 14:16] = False # keep hardware counters
        X_deg[drop] = 0.0
        mask[drop] = 0.0
        
    return X_deg, mask
