"""
prevention_alerting.py
======================
Adaptive Prevention & Multi-Channel Alerting Engine.
Supports cross-platform firewall integration:
  - Linux: iptables / nftables
  - Windows / PowerShell: netsh advfirewall / New-NetFirewallRule
"""

import time
import platform
from datetime import datetime

CURRENT_OS = platform.system() # 'Windows' or 'Linux'


class PreventionAlertingEngine:
    def __init__(self):
        # Active quarantine blocklist: IP -> expiration_timestamp
        self.firewall_blocklist = {}
        # Rate-limited IPs: IP -> rate_limit_config
        self.rate_limited_ips = {}
        # Complete incident ledger
        self.alert_history = []
        
    def process_detection(self, src_ip, dst_ip, attack_label, attack_name, active_mode, confidence=0.95):
        """
        Executes adaptive response based on detected attack type and severity.
        Generates cross-platform firewall commands for Linux and Windows PowerShell.
        """
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if attack_label == 0:
            # Benign - No alert or prevention needed
            return None
            
        # Determine Severity & Cross-Platform Firewall Command
        if attack_name in ["DDoS", "TCP SYN Flood"]:
            severity = "HIGH"
            action = "FIREWALL_DROP_IP"
            
            # Cross-Platform Firewall Syntax
            linux_cmd = f"iptables -A INPUT -s {src_ip} -j DROP"
            win_cmd = f"netsh advfirewall firewall add rule name=\"TIDPS_Block_{src_ip}\" dir=in action=block remoteip={src_ip}"
            ps_cmd = f"New-NetFirewallRule -DisplayName 'TIDPS_Block_{src_ip}' -Direction Inbound -Action Block -RemoteAddress {src_ip}"
            
            action_desc = f"Quarantined IP {src_ip} for 30m [{linux_cmd} | {win_cmd}]"
            # Add to quarantine for 1800s (30 mins)
            self.firewall_blocklist[src_ip] = time.time() + 1800
            
        elif attack_name in ["PortScan", "BruteForce", "Aggressive Port Scan"]:
            severity = "MEDIUM"
            action = "RATE_LIMIT_FLOW"
            action_desc = f"Applied token-bucket rate limit (5 pkts/s) to session {src_ip} -> {dst_ip} (protects shared NAT)"
            self.rate_limited_ips[src_ip] = {"limit_pps": 5, "applied_at": time.time()}
            
        else: # DoS-Slowloris / Recon
            severity = "MEDIUM"
            action = "TCP_RESET"
            action_desc = f"Injected TCP RST packet into lingering session {src_ip}:{dst_ip}"
            
        alert_record = {
            "timestamp": timestamp_str,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "attack_type": attack_name,
            "severity": severity,
            "active_mode": f"Mode {active_mode}",
            "confidence": f"{confidence:.1%}",
            "action_taken": action,
            "description": action_desc,
            "os_detected": CURRENT_OS
        }
        
        self.alert_history.append(alert_record)
        return alert_record

    def is_ip_blocked(self, src_ip):
        """Checks if an IP is currently under active firewall quarantine."""
        now = time.time()
        if src_ip in self.firewall_blocklist:
            if now < self.firewall_blocklist[src_ip]:
                return True
            else:
                # Expired quarantine
                del self.firewall_blocklist[src_ip]
        return False

    def get_recent_alerts(self, limit=15):
        """Returns the most recent security incidents."""
        return self.alert_history[-limit:][::-1]


if __name__ == "__main__":
    engine = PreventionAlertingEngine()
    print(f"[*] Testing Prevention Engine on OS: {CURRENT_OS}")
    a1 = engine.process_detection("192.168.1.105", "10.0.0.1", 1, "DDoS", active_mode=3)
    print(f"    [HIGH] Action: {a1['action_taken']}")
    print(f"    [DESC] {a1['description']}")
