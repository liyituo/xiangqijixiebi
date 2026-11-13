@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
title 工业级 9 点高精手眼标定程序
echo ========================================================
echo   正在启动【工业级 3x3 九点手眼标定系统】...
echo   消除广角镜头畸变，提升中心与边缘全场抓取精度！
echo ========================================================
"%PROJECT_PYTHON%" nine_point_calibration.py
pause
