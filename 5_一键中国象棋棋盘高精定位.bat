@echo off
chcp 65001 >nul
title 中国象棋棋盘高精全自动定位程序 - myCobot Pro 450
color 0b

echo ======================================================================
echo          🤖 中国象棋 9x10 棋盘一键视觉高精定位系统
echo          (支持 DashScope Qwen3.8-Flash + OpenCV 亚像素角点追踪)
echo ======================================================================
echo.
echo [提示] 正在调用相机拍照并进行棋盘四角 (a0, i0, i9, a9) 自动视觉标定...
echo [提示] 若网页控制大屏 (web_cockpit) 正在运行，定位完成后将自动热更新网格！
echo.

python xiangqi_auto_locate.py

echo.
echo ======================================================================
if %ERRORLEVEL% equ 0 (
    echo [成功] 棋盘定位已完成并自动保存至 xiangqi_board_config.json！
) else (
    echo [提示] 运行中出现异常，请检查相机连接或大模型 API 状态。
)
echo ======================================================================
echo.
pause
