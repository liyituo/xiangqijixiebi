@echo off
chcp 65001 >nul
title 大象机器人 x Qwen2.5-VL 视觉具身控制中枢
echo ========================================================
echo   正在启动【摄像头画面预览 + Qwen2.5-VL 视觉抓取客户端】...
echo ========================================================
python robot_vision_client.py
pause
