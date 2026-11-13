@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
title 大象机器人 x Qwen2.5-VL 视觉具身控制中枢
echo ========================================================
echo   正在启动【摄像头画面预览 + Qwen2.5-VL 视觉抓取客户端】...
echo ========================================================
"%PROJECT_PYTHON%" robot_vision_client.py
pause
