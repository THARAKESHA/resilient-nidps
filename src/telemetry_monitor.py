"""
telemetry_monitor.py
====================
Telemetry Quality Monitor (TQM).
Computes the normalized Telemetry Quality Vector q(t) = [C(t), S(t)]
and the composite Telemetry Quality Score Q(t) in [0.0, 1.0].
Categorizes telemetry health into:
  - GOOD (Q >= 0.80)
  - DEGRADED (0.40 <= Q < 0.80)
  - CRITICAL (Q < 0.40)
"""

import numpy as np
from src.data_loader import (
    FEATURE_NAMES_FULL,
    DEGRADED_INDICES,
    BACKWARD_FEATURE_INDICES,
    TIMING_FEATURE_INDICES
)

# Feature importance weights w_i based on mutual information in CIC-IDS2017
D = len(FEATURE_NAMES_FULL)
FEATURE_WEIGHTS = np.ones(D, dtype=np.float32)

# Higher weights assigned to critical discriminating features
for idx in DEGRADED_INDICES:
    FEATURE_WEIGHTS[idx] = 2.5
for idx in BACKWARD_FEATURE_INDICES:
    FEATURE_WEIGHTS[idx] = 2.0
for idx in TIMING_FEATURE_INDICES:
    FEATURE_WEIGHTS[idx] = 1.5

WEIGHT_SUM = np.sum(FEATURE_WEIGHTS)


def compute_completeness_index(mask):
    """
    Computes weighted feature completeness C in [0.0, 1.0].
    mask can be a 1D vector (single flow) or 2D matrix (batch).
    """
    if mask.ndim == 1:
        return float(np.sum(mask * FEATURE_WEIGHTS) / WEIGHT_SUM)
    else:
        return np.sum(mask * FEATURE_WEIGHTS, axis=1) / WEIGHT_SUM


def compute_consistency_index(X):
    """
    Evaluates adherence to fundamental physical network invariants:
      1. Flow Duration must be non-negative.
      2. If total packets > 0, total bytes must be >= packets * 20 (IP header minimum).
      3. Valid TCP destination port range (1 to 65535).
      4. Rate consistency: Byte rate <= 10 Gbps (1.25 GB/s wire physical maximum).
    Returns consistency score S in [0.0, 1.0].
    """
    if X.ndim == 1:
        X_mat = X.reshape(1, -1)
    else:
        X_mat = X
        
    N = len(X_mat)
    checks_passed = np.zeros(N, dtype=np.float32)
    total_checks = 4.0
    
    # Invariant 1: Positive Duration
    dur = X_mat[:, 1]
    checks_passed += (dur >= 0.0).astype(np.float32)
    
    # Invariant 2: Total bytes >= Packets * 20
    pkts = X_mat[:, 2] + X_mat[:, 3] # Fwd + Bwd
    bytes_tot = X_mat[:, 4] + X_mat[:, 5]
    valid_bytes = np.where(pkts > 0, bytes_tot >= (pkts * 20.0), True)
    checks_passed += valid_bytes.astype(np.float32)
    
    # Invariant 3: Destination Port range [1, 65535]
    dst_port = X_mat[:, 0]
    valid_port = (dst_port >= 0) & (dst_port <= 65535)
    checks_passed += valid_port.astype(np.float32)
    
    # Invariant 4: Physical wire rate limit (< 1.25 GB/s wire capacity)
    byte_rate = X_mat[:, 14]
    valid_rate = (byte_rate >= 0) & (byte_rate <= 1.25e9)
    checks_passed += valid_rate.astype(np.float32)
    
    S = checks_passed / total_checks
    return float(S[0]) if X.ndim == 1 else S


def assess_telemetry_quality(X, mask, alpha=0.70):
    """
    Computes composite Telemetry Quality Score Q in [0.0, 1.0].
    Q = alpha * C + (1 - alpha) * S
    Returns:
      Q: Composite score
      tier: 'GOOD', 'DEGRADED', or 'CRITICAL'
      C: Completeness score
      S: Consistency score
    """
    C = compute_completeness_index(mask)
    S = compute_consistency_index(X)
    Q = alpha * C + (1.0 - alpha) * S
    
    if np.isscalar(Q):
        if Q >= 0.80:
            tier = "GOOD"
        elif Q >= 0.40:
            tier = "DEGRADED"
        else:
            tier = "CRITICAL"
        return float(Q), tier, float(C), float(S)
    else:
        tiers = []
        for q_val in Q:
            if q_val >= 0.80:
                tiers.append("GOOD")
            elif q_val >= 0.40:
                tiers.append("DEGRADED")
            else:
                tiers.append("CRITICAL")
        return Q, np.array(tiers), C, S


if __name__ == "__main__":
    from src.data_loader import generate_calibrated_benchmark_dataset
    from src.degradation_simulator import apply_degradation
    
    X, _ = generate_calibrated_benchmark_dataset(num_samples=100)
    for lvl in [0, 1, 2, 3]:
        X_deg, mask = apply_degradation(X, level=lvl)
        Q, tier, C, S = assess_telemetry_quality(X_deg[0], mask[0])
        print(f"Level {lvl} -> Q: {Q:.3f} (C: {C:.3f}, S: {S:.3f}) | Quality Tier: {tier}")
