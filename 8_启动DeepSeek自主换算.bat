@echo off
chcp 65001 >nul
title DeepSeek-V4.1-Flash 自主视觉与空间物理坐标换算
echo ========================================================
echo   正在运行【DeepSeek-V4.1-Flash 4黑点自主空间换算求解器】...
echo   模型: deepseek-flash (多模态视觉)
echo ========================================================
python deepseek_workspace_solver.py
pause
