@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 first to install the isolated environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
pause
