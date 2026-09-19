@echo off
title Telemetry-Aware Resilient NIDPS Dashboard
echo =====================================================================
echo Starting Resilient AI-Based NIDPS with Telemetry-Aware Multi-Mode App...
echo =====================================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
