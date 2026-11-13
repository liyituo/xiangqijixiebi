#!/bin/bash
echo '=================================================='
echo '  正在启动 OpenVLA-7B 具身动作大模型服务 (RTX 4090)'
echo '=================================================='

# 1. 终止可能占用的旧服务
pkill -f app_openvla_api.py 2>/dev/null
pkill -f app_robot_vlm_api.py 2>/dev/null
pkill -f app_robot_api.py 2>/dev/null
sleep 1

# 2. 检查显卡状态
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# 3. 后台启动服务
nohup /root/miniconda3/bin/python /root/app_openvla_api.py > /root/openvla_api.log 2>&1 &
echo "OpenVLA 服务后台启动，PID: $!"

sleep 3
ps aux | grep app_openvla_api | grep -v grep

echo '--------------------------------------------------'
echo '服务将在 6006 端口提供 RESTful API:'
echo '  - 健康检查: GET  http://127.0.0.1:6006/health'
echo '  - 动作预测: POST http://127.0.0.1:6006/predict_action'
echo '  - 统一调度: POST http://127.0.0.1:6006/robot_vision_command'
echo '实时日志请查看: tail -f /root/openvla_api.log'
echo '=================================================='
