#!/bin/bash
# Telemetry-Aware Resilient NIDPS Linux Launcher
echo "====================================================================="
echo "Starting Resilient AI-Based NIDPS with Telemetry-Aware Multi-Mode App..."
echo "====================================================================="
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"
python3 -m streamlit run app.py
