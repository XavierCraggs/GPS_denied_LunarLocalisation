@echo off
REM Symmetry-check run for both scenes (launched by Claude). Logs -> results_silhouette\symcheck_*_log.txt
cd /d "%~dp0"
set SCENE=v1
.venv\Scripts\python.exe -u run_diagnostics.py symcheck > results_silhouette\symcheck_v1_log.txt 2>&1
set SCENE=v2
.venv\Scripts\python.exe -u run_diagnostics.py symcheck > results_silhouette\symcheck_v2_log.txt 2>&1
.venv\Scripts\python.exe -u run_diagnostics.py summary > results_silhouette\diag_summary_log.txt 2>&1
echo done > results_silhouette\symcheck_DONE.txt
