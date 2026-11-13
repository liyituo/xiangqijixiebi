@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
title DeepSeek-V4.1-Flash 全自主视觉手眼标定系统
echo ========================================================
echo   正在启动【DeepSeek-V4.1-Flash 全自主视觉手眼标定】...
echo   自动识别画面中 4 个黑点，无需人工手动点选！
echo ========================================================
"%PROJECT_PYTHON%" deepseek_auto_calibrate.py
pause
