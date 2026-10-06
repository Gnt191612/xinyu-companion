@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "心语樱花版.exe" (
  start "" "心语樱花版.exe"
) else (
  start "" ".venv\Scripts\pythonw.exe" "体验\desktop_app.py"
)
