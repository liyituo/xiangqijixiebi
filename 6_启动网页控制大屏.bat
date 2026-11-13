@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
title 机械臂多模态具身控制网页大屏 (Web Cockpit)
echo ========================================================
echo   正在启动【FlexArm 具身视觉网页大屏控制台】...
echo   服务地址: http://127.0.0.1:5000 (将自动在浏览器中打开)
echo ========================================================
"%PROJECT_PYTHON%" web_cockpit.py
pause
