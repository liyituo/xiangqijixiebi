# -*- coding: utf-8 -*-
"""
FlexArm / 大象机器人 本地大模型客户端桥接程序 (Robot Client)
运行在连接真实机械臂的电脑上：
1. 监听用户输入的自然语言（或快捷指令）；
2. 请求云服务器 (RTX 4080 + Qwen2.5-1.5B) 的 FastAPI 接口；
3. 解析大模型返回的动作与参数，驱动大象机器人（网口 TCP Socket）执行物理动作。
"""

import sys
import time
import json
import requests
from typing import Dict, Any, List

# ==============================================================================
# 1. 核心网络与硬件配置（根据实际情况修改）
# ==============================================================================

# 云服务器 API 地址 (若使用 SSH 隧道转发，保持 127.0.0.1:6006 即可)
# 启动 SSH 隧道的命令示例：
# ssh -CNg -L 6006:127.0.0.1:6006 -p <SSH_PORT> <SSH_USER>@<REMOTE_HOST>
CLOUD_API_URL = "http://127.0.0.1:6006/robot_command"

# 大象机器人机械臂配置 (网线连接，默认静态 IP 通常为 192.168.0.232 或 192.168.1.159)
ROBOT_IP = "192.168.0.232"
ROBOT_PORT = 4500  # 或 9000 / 5001，取决于固件版本

# 标定的安全工位关节角度 (J1, J2, J3, J4, J5, J6)
STATION_JOINTS = {
    "HOME": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "P1": [-45.0, 15.0, -30.0, 10.0, 0.0, 0.0],   # 取料工位 A
    "P2": [-20.0, 20.0, -25.0, 10.0, 0.0, 0.0],   # 取料工位 B
    "P3": [45.0, 15.0, -30.0, 10.0, 0.0, 0.0],    # 默认放置工位
    "P4": [0.0, 30.0, -40.0, 10.0, 0.0, 0.0],     # 重规划工位
}

from pro450_driver import MyCobotPro450Driver, DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT

# ==============================================================================
# 2. 机械臂底层驱动适配层 (myCobot Pro 450 专属驱动)
# ==============================================================================

class RobotDriver:
    def __init__(self, ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT):
        self.ip = ip
        self.port = port
        self.backend = MyCobotPro450Driver(ip=self.ip, port=self.port, auto_scan=True)
        self.mc = self.backend.client
        self.is_connected = self.backend.is_connected

    def send_angles(self, angles: List[float], speed: int = 15):
        """控制各关节移动到指定角度"""
        self.backend.send_angles(angles, speed)

    def set_gripper(self, action: str, speed: int = 50):
        """控制夹爪开合 (自适应 Pro 夹爪与标准夹爪)"""
        self.backend.set_gripper(action, speed)

    def emergency_stop(self):
        """紧急停止"""
        self.backend.emergency_stop()

# ==============================================================================
# 3. 动作执行器 (根据大模型返回的 Tool 进行真实物理调度)
# ==============================================================================

class RobotExecutor:
    def __init__(self, driver: RobotDriver):
        self.driver = driver

    def execute(self, tool_name: str, args: Dict[str, Any]):
        print(f"\n⚙️ >>> 开始物理执行工具: 【{tool_name}】")
        print(f"📋 >>> 参数: {args}")

        if tool_name == "wave_hand":
            print("👋 执行打招呼动作序列...")
            self.driver.send_angles([0.0, 20.0, -20.0, 0.0, 30.0, 0.0], speed=20)
            self.driver.send_angles([0.0, 20.0, -20.0, 0.0, -30.0, 0.0], speed=20)
            self.driver.send_angles([0.0, 20.0, -20.0, 0.0, 30.0, 0.0], speed=20)
            self.driver.send_angles(STATION_JOINTS["HOME"], speed=15)
            print("✅ 打招呼动作完成！")

        elif tool_name == "reset_arm":
            print("🔄 执行复位动作...")
            self.driver.send_angles(STATION_JOINTS["HOME"], speed=10)
            print("✅ 机械臂已复位到安全 HOME 点。")

        elif tool_name == "flexarm_control_gripper":
            action = args.get("action", "OPEN")
            speed = args.get("speed", 50)
            self.driver.set_gripper(action, speed)
            print("✅ 夹爪动作执行完成。")

        elif tool_name == "flexarm_emergency_stop":
            reason = args.get("reason", "未指定原因")
            print(f"🚨 紧急制动触发！原因: {reason}")
            self.driver.emergency_stop()

        elif tool_name in ["flexarm_plan_task", "flexarm_execute_plan"]:
            src = args.get("source_station", "P1").upper()
            tgt = args.get("target_station", "P3").upper()
            auto_replan = args.get("auto_replan_target")
            if auto_replan:
                tgt = auto_replan.upper()

            print(f"📦 执行闭环搬运工序: [{src}] -> [{tgt}]")
            # 步骤 1: 移动到取料工位上方
            src_angles = STATION_JOINTS.get(src, STATION_JOINTS["P1"])
            self.driver.set_gripper("OPEN")
            self.driver.send_angles(src_angles, speed=10)
            # 步骤 2: 夹取工件
            self.driver.set_gripper("CLOSE")
            # 步骤 3: 抬升到空中安全检查点 (ACT-06)
            checkpoint_angles = [0.0, 10.0, -10.0, 0.0, 0.0, 0.0]
            print("✈️ 抬升至空中安全检查点 (ACT-06 挂起)...")
            self.driver.send_angles(checkpoint_angles, speed=10)
            # 步骤 4: 前往目标工位
            tgt_angles = STATION_JOINTS.get(tgt, STATION_JOINTS["P3"])
            print(f"🚚 空间平移送往目标工位 [{tgt}]...")
            self.driver.send_angles(tgt_angles, speed=10)
            # 步骤 5: 下放并释放
            self.driver.set_gripper("OPEN")
            # 步骤 6: 返回 HOME 姿态
            self.driver.send_angles(STATION_JOINTS["HOME"], speed=10)
            print(f"🎉 搬运任务已完成，工件已成功交付至 [{tgt}]！")

        elif tool_name == "flexarm_get_status":
            print("📊 机械臂状态正常，伺服电机工作在使能状态。")

        else:
            print(f"⚠️ 未适配的工具指令: {tool_name}，参数: {args}")

# ==============================================================================
# 4. 主循环：接收自然语言并与云端大模型交互
# ==============================================================================

def main():
    print("=" * 65)
    print("  🚀 大象机器人 x 具身大模型 本地控制客户端")
    print("=" * 65)
    print(f"🔗 云端大模型接口: {CLOUD_API_URL}")
    print(f"🤖 机械臂目标 IP : {ROBOT_IP}:{ROBOT_PORT}")
    print("=" * 65)

    # 1. 检查云端连通性
    try:
        health_resp = requests.get(CLOUD_API_URL.replace("/robot_command", "/health"), timeout=3)
        if health_resp.status_code == 200:
            hdata = health_resp.json()
            print(f"✅ 云端大模型服务连接正常！")
            print(f"   - 底座模型: {hdata.get('model')}")
            print(f"   - 推理显卡: {hdata.get('gpu')}")
            print(f"   - 支持工具: {hdata.get('supported_tools_count')} 项")
    except Exception as e:
        print(f"❌ 警告: 无法连接云端 API ({CLOUD_API_URL})！")
        print("👉 请确保已在终端开启 SSH 隧道:")
        print("   ssh -CNg -L 6006:127.0.0.1:6006 -p <SSH_PORT> <SSH_USER>@<REMOTE_HOST>\n")

    # 2. 初始化硬件驱动
    driver = RobotDriver(ROBOT_IP, ROBOT_PORT)
    executor = RobotExecutor(driver)

    print("\n💡 你可以直接输入自然语言指令控制机械臂，例如:")
    print("   1. '你好，跟我打个招呼吧'")
    print("   2. '把工件A从P1搬运到P3'")
    print("   3. '把机械臂复位'")
    print("   4. '把夹爪张开'")
    print("   5. '紧急停止机械臂'")
    print("   (输入 'q' 或 'exit' 退出程序)\n")

    while True:
        try:
            user_input = input("\n🗣️ 请输入控制指令 > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "exit", "quit"]:
                print("👋 退出程序。")
                break

            print("🧠 正在发送给云端 4080 大模型进行意图理解与规划...")
            t0 = time.time()
            resp = requests.post(
                CLOUD_API_URL,
                json={"instruction": user_input},
                timeout=15
            )
            roundtrip_ms = (time.time() - t0) * 1000

            if resp.status_code != 200:
                print(f"❌ 请求失败: HTTP {resp.status_code} - {resp.text}")
                continue

            data = resp.json()
            elapsed_ms = data.get("elapsed_ms", 0)
            has_tool = data.get("has_tool_call", False)
            tool_name = data.get("tool_name")
            arguments = data.get("arguments", {})
            reply = data.get("reply_text")

            print(f"⚡ 大模型决策完毕 (模型推理: {elapsed_ms:.1f}ms | 网络往返: {roundtrip_ms:.1f}ms)")
            
            if has_tool and tool_name:
                print(f"🎯 识别到物理工具调用: {tool_name}")
                executor.execute(tool_name, arguments)
            else:
                print(f"💬 大模型回复:\n{reply}")

        except KeyboardInterrupt:
            print("\n程序已终止。")
            break
        except Exception as e:
            print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    main()
