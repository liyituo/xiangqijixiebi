@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -B detect_robot.py
pause
