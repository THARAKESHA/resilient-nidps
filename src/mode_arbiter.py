"""
mode_arbiter.py
===============
Dynamic Mode Arbiter with Hysteresis Stabilization.
Manages prevalidated detection mode dispatch and eliminates high-frequency
mode flapping under fluctuating telemetry boundaries.

Operating Modes:
  - Mode 1: Full-Feature ML (Q >= 0.80)
  - Mode 2: Reduced L3/L4 ML (0.40 <= Q < 0.80)
  - Mode 3: Fallback Rule Engine (Q < 0.40)
"""

import numpy as np


class ModeArbiter:
    def __init__(self, tau_high=0.80, tau_low=0.40, delta=0.05):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.delta = delta # Hysteresis band
        self.current_mode = 1 # Start in Mode 1 (Full)
        self.switch_count = 0
        self.flapping_prevented_count = 0
        
    def select_mode(self, Q):
        """
        Determines the appropriate detection mode based on Quality Score Q
        incorporating hysteresis to prevent mode flapping.
        """
        prev_mode = self.current_mode
        new_mode = prev_mode
        
        if prev_mode == 1:
            # Currently in Mode 1 (Full)
            # Do NOT drop to Mode 2 unless Q falls below (tau_high - delta)
            if Q < (self.tau_high - self.delta):
                if Q < self.tau_low:
                    new_mode = 3 # Sudden crash directly to Fallback
                else:
                    new_mode = 2
            elif Q < self.tau_high:
                # Flapping prevented!
                self.flapping_prevented_count += 1
                
        elif prev_mode == 2:
            # Currently in Mode 2 (Degraded)
            if Q >= self.tau_high:
                new_mode = 1 # Promoted back to Full
            elif Q < (self.tau_low - self.delta):
                new_mode = 3 # Dropped to Fallback
            elif Q < self.tau_low:
                # Flapping prevented on lower boundary!
                self.flapping_prevented_count += 1
                
        elif prev_mode == 3:
            # Currently in Mode 3 (Fallback)
            if Q >= self.tau_high:
                new_mode = 1 # Complete recovery
            elif Q >= self.tau_low:
                new_mode = 2 # Partial recovery to Degraded
            elif Q >= (self.tau_low - self.delta):
                # Flapping prevented!
                self.flapping_prevented_count += 1
                
        if new_mode != prev_mode:
            self.switch_count += 1
            self.current_mode = new_mode
            
        return new_mode

    def arbitrate_batch(self, Q_array):
        """
        Arbitrates a sequence of flows, tracking transitions and flapping.
        """
        modes = []
        for q in Q_array:
            m = self.select_mode(q)
            modes.append(m)
        return np.array(modes)

    def reset(self):
        self.current_mode = 1
        self.switch_count = 0
        self.flapping_prevented_count = 0


if __name__ == "__main__":
    arbiter = ModeArbiter(tau_high=0.80, tau_low=0.40, delta=0.05)
    
    # Simulate a noisy boundary oscillation: 0.81 -> 0.78 -> 0.82 -> 0.77 -> 0.38 -> 0.33
    test_Q_stream = [0.95, 0.82, 0.78, 0.81, 0.77, 0.74, 0.55, 0.42, 0.38, 0.32, 0.45, 0.85]
    print("[*] Simulating Dynamic Mode Arbitration with Hysteresis:")
    for q in test_Q_stream:
        mode = arbiter.select_mode(q)
        print(f"    Q = {q:.2f} -> Active Detection Mode: Mode {mode}")
        
    print(f"[*] Total Switches: {arbiter.switch_count}")
    print(f"[*] Mode Flapping Events Prevented by Hysteresis: {arbiter.flapping_prevented_count}")
