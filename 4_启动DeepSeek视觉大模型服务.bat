@echo off
chcp 65001 >nul
title FlexArm x Qwen3.8-Flash 视觉大模型服务
echo ========================================================
echo   正在启动【Qwen3.8-Flash 极速具身视觉感知服务】...
echo   模型: qwen3.8-flash (百炼 Model Studio 多模态视觉)
echo   端点: http://127.0.0.1:6006
echo ========================================================
python app_deepseek_vlm_service.py
pause
