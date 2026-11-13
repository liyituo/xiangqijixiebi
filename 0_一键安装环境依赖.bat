@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 项目环境配置
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_environment.ps1" -Mode All
if errorlevel 1 (
    echo 环境配置失败，请查看上方错误。
) else (
    echo 本地控制、视觉模型与 OpenVLA 兼容环境配置完成。
)
pause
