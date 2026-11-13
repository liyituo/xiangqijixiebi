@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
title 大象机器人 x 具身大模型控制台
echo ========================================================
echo   正在启动机械臂交互控制台...
echo ========================================================
"%PROJECT_PYTHON%" robot_client.py
pause
