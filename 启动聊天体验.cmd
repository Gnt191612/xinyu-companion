@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在加载模型，显示入口已就绪后，请打开 http://127.0.0.1:7860
".venv\Scripts\python.exe" "体验\chat_app.py"
pause
