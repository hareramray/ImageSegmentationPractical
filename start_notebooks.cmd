@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project Python environment is missing. Follow README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m jupyterlab --notebook-dir="." --LabApp.default_url="/lab/tree/notebooks/00_start_here.ipynb"
