@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
title DeepSeek-V4.1-Flash 自主视觉与空间物理坐标换算
echo ========================================================
echo   正在运行【DeepSeek-V4.1-Flash 4黑点自主空间换算求解器】...
echo   模型: deepseek-flash (多模态视觉)
echo ========================================================
"%PROJECT_PYTHON%" deepseek_workspace_solver.py
pause
