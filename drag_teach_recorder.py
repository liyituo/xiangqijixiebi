# -*- coding: utf-8 -*-
"""
myCobot Pro 450 自由拖动示教与点位录制中枢 (Drag-Teach & Waypoint Recorder)
适用型号: Elephant Robotics myCobot Pro 450 (网口 TCP Socket 通信)

核心亮点：
1. 【零重力低阻尼自由拖动】：调用 Pro 450 专属底层 set_free_move_mode(1) 降低阻尼，用手随意拖拽调整位姿；
2. 【一键回车秒级记录】：用户把机械臂移动到目标姿态，直接按下【回车 Enter】即可完整捕获 J1~J6 关节角度与笛卡尔空间坐标；
3. 【自适应夹爪无缝同步】：输入 'g' 一键切换夹爪张开/闭合状态，自动适配 Pro 夹爪与标准夹爪；
4. 【多维工业格式归档】：自动输出时间戳 JSON、latest_trajectory.json 与 Excel CSV 文件，包含毫米与角度数据；
5. 【实时真机复现回放】：录制中随时输入 'r'，或录制结束后一键在真机上高保真回放整个动作流；
6. 【硬件安全防下坠机制】：强制进入/退出扭矩管理闭环，退出程序时 100% 自动锁死刚性保持。
"""

import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from pro450_driver import MyCobotPro450Driver, DEFAULT_ROBOT_IP, DEFAULT_ROBOT_PORT, PRO450_LIMITS


class DragTeachRecorderApp:
    """myCobot Pro 450 拖动示教交互录制中枢"""

    def __init__(self, ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT):
        self.ip = ip
        self.port = port
        self.driver = MyCobotPro450Driver(ip=self.ip, port=self.port, auto_scan=True)
        self.waypoints: List[Dict[str, Any]] = []
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")
        os.makedirs(self.output_dir, exist_ok=True)

    def print_banner(self):
        print("\n" + "=" * 74)
        print("   🤖 Elephant Robotics myCobot Pro 450 自由拖动示教与点位录制工具")
        print("=" * 74)
        status_text = f"真机在线 ({self.driver.driver_type} @ {self.driver.ip}:{self.driver.port})" if self.driver.is_connected else "高保真物理仿真模式"
        print(f"  当前通信状态 : {status_text}")
        print("  核心交互指南:")
        print("  1. 阻尼降低后，请直接【用手轻扶/握住机械臂】移动至期望位姿")
        print("  2. 到位后，直接按【回车 Enter】即可保存当前点位")
        print("  3. 快捷指令按键:")
        print("     - 【直接按回车】   : 记录当前姿态为点位 (默认编号 P1, P2...)")
        print("     - 【输入自定义名】 : 如输入 PICK、PLACE、DROP 直接记录带标签点位")
        print("     - 【g 或 grip】    : 切换夹爪开合 (全开 100% / 夹紧 0%)")
        print("     - 【d 或 del】     : 撤销删除最近录制的一个点位")
        print("     - 【l 或 list】    : 打印当前所有已录制的点位清单表格")
        print("     - 【r 或 replay】  : 实时锁紧舵机并高保真试跑回放一次当前轨迹")
        print("     - 【q 或 save】    : 结束录制，锁紧舵机并导出轨迹文件")
        print("=" * 74 + "\n")

    def run(self):
        self.print_banner()

        # 1. 极其重要的物理安全防撞提醒
        print("⚠️ 【物理安全防跌落提醒】:")
        print("   进入低阻尼零重力自由拖动后，各轴关节电机关节扭矩卸载，保持力降低！")
        print("   👉 请在按下回车前，【务必用手轻托住机械臂小臂或末端】，防止重力坠碰桌面！")
        input("\n👉 准备好后，请按【回车 Enter】降低阻尼，开启自由拖拽: ")

        # 2. 降低阻尼
        self.driver.enable_free_drag()

        point_idx = 1
        try:
            while True:
                prompt_text = f"\n[点位 {point_idx}] 移动到位后按【回车】记录 (或输入标签/指令): "
                user_input = input(prompt_text).strip()

                # 指令分支: 保存退出
                if user_input.lower() in ["q", "quit", "exit", "done", "s", "save"]:
                    if len(self.waypoints) == 0:
                        confirm = input("⚠️ 当前尚未记录任何点位，确认退出吗？(y/n): ").strip().lower()
                        if confirm in ["y", "yes"]:
                            break
                        continue
                    else:
                        break

                # 指令分支: 切换夹爪
                elif user_input.lower() in ["g", "grip", "gripper"]:
                    current_grip = self.driver.get_gripper_value()
                    new_val = 0 if current_grip > 50 else 100
                    self.driver.set_gripper_value(new_val)
                    time.sleep(0.3)
                    continue

                # 指令分支: 撤销删除上一点
                elif user_input.lower() in ["d", "del", "delete"]:
                    if self.waypoints:
                        popped = self.waypoints.pop()
                        point_idx = max(1, point_idx - 1)
                        print(f"🗑️ 已撤销删除点位: 【{popped['name']}】 (当前剩余 {len(self.waypoints)} 个点位)")
                    else:
                        print("⚠️ 当前点位列表为空，无法撤销。")
                    continue

                # 指令分支: 查看清单表格
                elif user_input.lower() in ["l", "list"]:
                    self.show_point_list()
                    continue

                # 指令分支: 试跑回放
                elif user_input.lower() in ["r", "replay"]:
                    self.replay_current_sequence()
                    # 回放后继续恢复拖拽
                    input("\n👉 恢复自由拖拽模式，请轻托住机械臂后按【回车 Enter】继续: ")
                    self.driver.enable_free_drag()
                    continue

                # 正常点位记录 (回车或自定义命名)
                point_name = user_input if user_input else f"P{point_idx}"
                
                # 读取硬件当前姿态
                angles = self.driver.get_angles()
                coords = self.driver.get_coords()
                grip_val = self.driver.get_gripper_value()

                record_item = {
                    "index": point_idx,
                    "name": point_name,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "joint_angles_deg": angles,
                    "cartesian_coords_mm": coords,
                    "gripper_percent": grip_val
                }
                self.waypoints.append(record_item)

                # 格式化终端输出
                grip_desc = "张开 100%" if grip_val > 50 else f"开度 {grip_val}%" if grip_val > 0 else "夹紧 0%"
                print(f"✅ 【已成功记录点位 {point_idx}】 -> 标签名称: 【{point_name}】")
                print(f"   🔩 6轴关节角度 (J1~J6) : {angles}")
                print(f"   📐 笛卡尔空间坐标(XYZ/R): X={coords[0]:.1f}mm, Y={coords[1]:.1f}mm, Z={coords[2]:.1f}mm | Rx={coords[3]:.1f}°, Ry={coords[4]:.1f}°, Rz={coords[5]:.1f}°")
                print(f"   🤏 夹爪行程状态       : {grip_desc}")

                point_idx += 1

        except KeyboardInterrupt:
            print("\n\n🛑 检测到中断信号 (Ctrl+C)，正在安全锁紧舵机...")

        finally:
            # 无论何种情况退出，必须彻底锁紧舵机，保护硬件
            self.driver.disable_free_drag()

        # 3. 保存文件并提示回放
        if self.waypoints:
            self.save_trajectories()
            self.show_point_list()
            
            ans = input("\n🎉 录制完成！是否立即按照已录制轨迹在机械臂上进行慢速复现回放？(Y/n): ").strip().lower()
            if ans in ["", "y", "yes"]:
                self.replay_current_sequence()

        print("\n👋 myCobot Pro 450 拖动示教录制中枢已安全退出，谢谢使用！\n")

    def show_point_list(self):
        """展示所有已记录点位的表格"""
        if not self.waypoints:
            print("📋 当前尚未记录任何点位。")
            return
        print("\n" + "-" * 78)
        print(f"📋 myCobot Pro 450 已录制点位清单 (共 {len(self.waypoints)} 个航点):")
        print(f"{'序号':<6}{'标签名称':<12}{'时间':<10}{'笛卡尔空间 (X, Y, Z mm)':<26}{'夹爪开度'}")
        print("-" * 78)
        for wp in self.waypoints:
            c = wp["cartesian_coords_mm"]
            c_str = f"({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})"
            g_str = f"{wp['gripper_percent']}%"
            print(f"{wp['index']:<6}{wp['name']:<12}{wp['timestamp']:<10}{c_str:<26}{g_str}")
        print("-" * 78)

    def save_trajectories(self):
        """保存轨迹为 JSON 与 CSV 格式"""
        t_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(self.output_dir, f"pro450_trajectory_{t_stamp}.json")
        latest_file = os.path.join(self.output_dir, "latest_trajectory.json")
        csv_file = os.path.join(self.output_dir, f"pro450_trajectory_{t_stamp}.csv")

        export_data = {
            "robot_model": "Elephant Robotics myCobot Pro 450",
            "recorded_at": datetime.now().isoformat(),
            "total_waypoints": len(self.waypoints),
            "waypoints": self.waypoints
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)

        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4, ensure_ascii=False)

        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["序号", "点位名称", "录制时间", "J1(°)", "J2(°)", "J3(°)", "J4(°)", "J5(°)", "J6(°)", "X(mm)", "Y(mm)", "Z(mm)", "Rx(°)", "Ry(°)", "Rz(°)", "夹爪开度(%)"])
            for wp in self.waypoints:
                j = wp["joint_angles_deg"]
                c = wp["cartesian_coords_mm"]
                writer.writerow([wp["index"], wp["name"], wp["timestamp"], j[0], j[1], j[2], j[3], j[4], j[5], c[0], c[1], c[2], c[3], c[4], c[5], wp["gripper_percent"]])

        print(f"\n💾 轨迹文件已成功保存:")
        print(f"   - JSON 完整档案 : {json_file}")
        print(f"   - 最新调用通道   : {latest_file}")
        print(f"   - CSV 电子表格   : {csv_file}")

    def replay_current_sequence(self):
        """真机复现回放已录制的动作流"""
        if not self.waypoints:
            print("⚠️ 没有可回放的点位。")
            return

        print("\n🚀 >>> 开始在 myCobot Pro 450 上复现回放轨迹...")
        # 确保退出自由拖动，舵机扭矩恢复
        self.driver.disable_free_drag()
        time.sleep(0.5)

        for i, wp in enumerate(self.waypoints):
            print(f"   ▶ [{i+1}/{len(self.waypoints)}] 正在移动至 -> 【{wp['name']}】 ...")
            # 1. 设置夹爪
            self.driver.set_gripper_value(wp["gripper_percent"], speed=60)
            # 2. 平滑运行关节角度
            self.driver.send_angles(wp["joint_angles_deg"], speed=15)
            time.sleep(0.6)

        print("🎉 myCobot Pro 450 轨迹复现回放圆满完成！\n")


def parse_args():
    parser = argparse.ArgumentParser(description="myCobot Pro 450 Drag-Teach Recorder")
    parser.add_argument("--ip", type=str, default=DEFAULT_ROBOT_IP, help=f"Robot IP (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_ROBOT_PORT, help=f"Robot Port (default: {DEFAULT_ROBOT_PORT})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = DragTeachRecorderApp(ip=args.ip, port=args.port)
    app.run()
