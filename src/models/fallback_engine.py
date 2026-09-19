"""
fallback_engine.py
==================
Mode 3: Deterministic Heuristic and Threshold Rule Engine.
Operates in < 5 microseconds using coarse volumetric hardware signals.
"""

import numpy as np
from src.data_loader import ATTACK_CLASSES


class FallbackRuleEngine:
    def __init__(self, pps_threshold=20000.0, bps_threshold=10e6, syn_ratio_threshold=5.0):
        self.pps_threshold = pps_threshold
        self.bps_threshold = bps_threshold
        self.syn_ratio_threshold = syn_ratio_threshold
        
    def predict(self, X_full):
        """
        Infers attack labels from volumetric invariants.
        Returns array of class predictions: 0=Benign, 1=DDoS, 2=PortScan, 3=Slowloris, 4=BruteForce.
        """
        if X_full.ndim == 1:
            X_mat = X_full.reshape(1, -1)
        else:
            X_mat = X_full
            
        N = len(X_mat)
        preds = np.zeros(N, dtype=np.int32)
        
        dst_port = X_mat[:, 0]
        byte_rate = X_mat[:, 14]
        pkt_rate = X_mat[:, 15]
        syn_count = X_mat[:, 44]
        ack_count = X_mat[:, 47]
        
        # Rule 1: High Volumetric Surge (DDoS)
        ddos_mask = (pkt_rate > self.pps_threshold) | (byte_rate > self.bps_threshold)
        preds[ddos_mask] = 1 # DDoS
        
        # Rule 2: SYN Flood (Heavy SYN with zero/low ACK)
        syn_flood = (syn_count > 10) & ((syn_count / (ack_count + 1.0)) > self.syn_ratio_threshold)
        preds[syn_flood] = 1
        
        # Rule 3: Single probe packets to ephemeral/scanned ports (PortScan)
        port_scan = (pkt_rate < 5000) & (syn_count >= 1) & (ack_count == 0) & (dst_port > 1024)
        preds[port_scan & ~ddos_mask] = 2 # PortScan
        
        return preds[0] if X_full.ndim == 1 else preds

    def predict_proba(self, X_full):
        """Returns pseudo-confidence probabilities for decision consistency."""
        preds = self.predict(X_full)
        if np.isscalar(preds) or preds.ndim == 0:
            preds = np.array([preds])
        probas = np.zeros((len(preds), len(ATTACK_CLASSES)), dtype=np.float32)
        for i, p in enumerate(preds):
            probas[i, p] = 0.95
            probas[i, (p + 1) % len(ATTACK_CLASSES)] = 0.05
        return probas
