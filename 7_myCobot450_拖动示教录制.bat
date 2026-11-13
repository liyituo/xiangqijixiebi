@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo 请先运行 0_一键安装环境依赖.bat
    pause
    exit /b 1
)
title myCobot Pro 450 零重力拖动示教与点位录制
echo ========================================================
echo   🤖 myCobot Pro 450 机械臂自由拖动示教与点位录制程序
echo ========================================================
echo.
echo 提示: 默认连接 IP 为 192.168.0.232 (端口 4500)
echo 若使用默认 IP，直接按【回车 Enter】即可启动；
echo 若机械臂配置为其他局域网 IP，请输入后按回车:
set /p ROBOT_IP_INPUT="👉 请输入机械臂 IP [默认 192.168.0.232]: "
if "%ROBOT_IP_INPUT%"=="" set ROBOT_IP_INPUT=192.168.0.232
echo.
echo 🚀 正在连接 myCobot Pro 450 (%ROBOT_IP_INPUT%:4500)...
echo.
"%PROJECT_PYTHON%" drag_teach_recorder.py --ip %ROBOT_IP_INPUT%
pause
