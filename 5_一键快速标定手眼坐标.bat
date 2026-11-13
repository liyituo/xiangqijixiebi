@echo off
chcp 65001 >nul
title DeepSeek-V4.1-Flash 全自主视觉手眼标定系统
echo ========================================================
echo   正在启动【DeepSeek-V4.1-Flash 全自主视觉手眼标定】...
echo   自动识别画面中 4 个黑点，无需人工手动点选！
echo ========================================================
python deepseek_auto_calibrate.py
pause
