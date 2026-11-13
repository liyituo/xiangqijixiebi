# -*- coding: utf-8 -*-
"""
myCobot Pro 450 自由拖动示教与点位录制中枢 (Drag-Teach & Waypoint Recorder)
适用型号: Elephant Robotics myCobot Pro 450 (网口 TCP Socket 通信)
"""

import os
import sys
import time
import csv
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from json_store import atomic_write_json
from pro450_driver import (
    MyCobotPro450Driver,
    DEFAULT_ROBOT_IP,
    DEFAULT_ROBOT_PORT,
    is_valid_vector,
)

GRIPPER_OPEN_PERCENT = 15
GRIPPER_CLOSE_PERCENT = 5


class DragTeachRecorderApp:
    """myCobot Pro 450 拖动示教交互录制中枢"""

    def __init__(self, ip: str = DEFAULT_ROBOT_IP, port: int = DEFAULT_ROBOT_PORT, auto_connect: bool = True):
        self.ip = ip
        self.port = port
        self.driver = (
            MyCobotPro450Driver(ip=self.ip, port=self.port, auto_scan=True)
            if auto_connect
            else None
        )
        self.waypoints: List[Dict[str, Any]] = []
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")
        os.makedirs(self.output_dir, exist_ok=True)

    def _read_live_pose(self):
        """使用 allow_cache=False 读取实时硬件位姿与夹爪开度。"""
        try:
            angles = self.driver.get_angles(allow_cache=False)
        except TypeError:
            angles = self.driver.get_angles()
        try:
            coords = self.driver.get_coords(allow_cache=False)
        except TypeError:
            coords = self.driver.get_coords()
        try:
            grip_val = self.driver.get_gripper_value(allow_cache=False)
        except TypeError:
            grip_val = self.driver.get_gripper_value()
        return angles, coords, grip_val

    def record_current_waypoint(self, point_name: str = "") -> Optional[Dict[str, Any]]:
        """捕获并校验当前实时位姿后加入 waypoints。"""
        angles, coords, grip_val = self._read_live_pose()
        if not is_valid_vector(angles, 6) or not is_valid_vector(coords, 6):
            print("❌ 未读取到有效机械臂实时位姿（angles/coords 无效或通信异常），拒绝记录坏点位。")
            return None

        idx = len(self.waypoints) + 1
        name = point_name.strip() if point_name and point_name.strip() else f"P{idx}"
        grip_int = int(round(grip_val)) if isinstance(grip_val, (int, float)) and not isinstance(grip_val, bool) else GRIPPER_OPEN_PERCENT
        record_item = {
            "index": idx,
            "name": name,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "joint_angles_deg": [float(v) for v in angles[:6]],
            "cartesian_coords_mm": [float(v) for v in coords[:6]],
            "gripper_percent": grip_int,
        }
        self.waypoints.append(record_item)
        return record_item

    def print_banner(self):
        print("\n" + "=" * 74)
        print("   🤖 Elephant Robotics myCobot Pro 450 自由拖动示教与点位录制工具")
        print("=" * 74)
        status_text = (
            f"真机在线 ({self.driver.driver_type} @ {self.driver.ip}:{self.driver.port})"
            if self.driver and self.driver.is_connected
            else "离线/仿真模式"
        )
        print(f"  当前通信状态 : {status_text}")
        print("  核心交互指南:")
        print("  1. 阻尼降低后，请直接【用手轻扶/握住机械臂】移动至期望位姿")
        print("  2. 到位后，直接按【回车 Enter】即可保存当前点位")
        print("  3. 快捷指令按键:")
        print("     - 【直接按回车】   : 记录当前姿态为点位 (默认编号 P1, P2...)")
        print("     - 【输入自定义名】 : 如输入 PICK、PLACE、DROP 直接记录带标签点位")
        print(f"     - 【g 或 grip】    : 切换夹爪开合 (张开 {GRIPPER_OPEN_PERCENT}% / 夹紧 {GRIPPER_CLOSE_PERCENT}%)")
        print("     - 【d 或 del】     : 撤销删除最近录制的一个点位")
        print("     - 【l 或 list】    : 打印当前所有已录制的点位清单表格")
        print("     - 【r 或 replay】  : 实时锁紧舵机并高保真试跑回放一次当前轨迹")
        print("     - 【q 或 save】    : 结束录制，锁紧舵机并导出轨迹文件")
        print("=" * 74 + "\n")

    def run(self):
        self.print_banner()

        print("⚠️ 【物理安全防跌落提醒】:")
        print("   进入低阻尼零重力自由拖动后，各轴关节电机关节扭矩卸载，保持力降低！")
        print("   👉 请在按下回车前，【务必用手轻托住机械臂小臂或末端】，防止重力坠碰桌面！")
        input("\n👉 准备好后，请按【回车 Enter】降低阻尼，开启自由拖拽: ")

        self.driver.enable_free_drag()

        try:
            while True:
                next_idx = len(self.waypoints) + 1
                prompt_text = f"\n[点位 {next_idx}] 移动到位后按【回车】记录 (或输入标签/指令): "
                user_input = input(prompt_text).strip()

                if user_input.lower() in ["q", "quit", "exit", "done", "s", "save"]:
                    if len(self.waypoints) == 0:
                        confirm = input("⚠️ 当前尚未记录任何点位，确认退出吗？(y/n): ").strip().lower()
                        if confirm in ["y", "yes"]:
                            break
                        continue
                    break

                elif user_input.lower() in ["g", "grip", "gripper"]:
                    try:
                        current_grip = self.driver.get_gripper_value(allow_cache=False)
                    except TypeError:
                        current_grip = self.driver.get_gripper_value()
                    threshold = (GRIPPER_OPEN_PERCENT + GRIPPER_CLOSE_PERCENT) / 2.0
                    new_val = (
                        GRIPPER_CLOSE_PERCENT
                        if (isinstance(current_grip, (int, float)) and current_grip > threshold)
                        else GRIPPER_OPEN_PERCENT
                    )
                    self.driver.set_gripper_value(new_val)
                    time.sleep(0.3)
                    continue

                elif user_input.lower() in ["d", "del", "delete"]:
                    if self.waypoints:
                        popped = self.waypoints.pop()
                        print(f"🗑️ 已撤销删除点位: 【{popped['name']}】 (当前剩余 {len(self.waypoints)} 个点位)")
                    else:
                        print("⚠️ 当前点位列表为空，无法撤销。")
                    continue

                elif user_input.lower() in ["l", "list"]:
                    self.show_point_list()
                    continue

                elif user_input.lower() in ["r", "replay"]:
                    self.replay_current_sequence()
                    input("\n👉 恢复自由拖拽模式，请轻托住机械臂后按【回车 Enter】继续: ")
                    self.driver.enable_free_drag()
                    continue

                record_item = self.record_current_waypoint(user_input)
                if record_item is None:
                    continue

                angles = record_item["joint_angles_deg"]
                coords = record_item["cartesian_coords_mm"]
                grip_val = record_item["gripper_percent"]
                grip_desc = f"张开 ({grip_val}%)" if grip_val > GRIPPER_CLOSE_PERCENT else f"夹紧 ({grip_val}%)"
                print(f"✅ 【已成功记录点位 {record_item['index']}】 -> 标签名称: 【{record_item['name']}】")
                print(f"   🔩 6轴关节角度 (J1~J6) : {angles}")
                print(
                    f"   📐 笛卡尔空间坐标(XYZ/R): X={coords[0]:.1f}mm, Y={coords[1]:.1f}mm, "
                    f"Z={coords[2]:.1f}mm | Rx={coords[3]:.1f}°, Ry={coords[4]:.1f}°, Rz={coords[5]:.1f}°"
                )
                print(f"   🤏 夹爪行程状态       : {grip_desc}")

        except KeyboardInterrupt:
            print("\n\n🛑 检测到中断信号 (Ctrl+C)，正在安全锁紧舵机...")

        finally:
            if self.driver is not None:
                self.driver.disable_free_drag()

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
        """原子写入保存轨迹为 JSON 与 CSV 格式"""
        t_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(self.output_dir, f"pro450_trajectory_{t_stamp}.json")
        latest_file = os.path.join(self.output_dir, "latest_trajectory.json")
        csv_file = os.path.join(self.output_dir, f"pro450_trajectory_{t_stamp}.csv")

        snapshot = list(self.waypoints)
        export_data = {
            "robot_model": "Elephant Robotics myCobot Pro 450",
            "recorded_at": datetime.now().isoformat(),
            "total_waypoints": len(snapshot),
            "waypoints": snapshot,
        }

        atomic_write_json(json_file, export_data, indent=4)
        atomic_write_json(latest_file, export_data, indent=4)

        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "序号", "点位名称", "录制时间",
                    "J1(°)", "J2(°)", "J3(°)", "J4(°)", "J5(°)", "J6(°)",
                    "X(mm)", "Y(mm)", "Z(mm)", "Rx(°)", "Ry(°)", "Rz(°)", "夹爪开度(%)",
                ]
            )
            for wp in snapshot:
                j = wp["joint_angles_deg"]
                c = wp["cartesian_coords_mm"]
                writer.writerow(
                    [
                        wp["index"], wp["name"], wp["timestamp"],
                        j[0], j[1], j[2], j[3], j[4], j[5],
                        c[0], c[1], c[2], c[3], c[4], c[5], wp["gripper_percent"],
                    ]
                )

        print("\n💾 轨迹文件已成功保存:")
        print(f"   - JSON 完整档案 : {json_file}")
        print(f"   - 最新调用通道   : {latest_file}")
        print(f"   - CSV 电子表格   : {csv_file}")

    def replay_current_sequence(self) -> bool:
        """真机复现回放已录制的动作流"""
        if not self.waypoints:
            print("⚠️ 没有可回放的点位。")
            return False

        snapshot = list(self.waypoints)
        print("\n🚀 >>> 开始在 myCobot Pro 450 上复现回放轨迹...")
        self.driver.disable_free_drag()
        time.sleep(0.5)

        for i, wp in enumerate(snapshot):
            print(f"   ▶ [{i+1}/{len(snapshot)}] 正在移动至 -> 【{wp['name']}】 ...")
            if not self.driver.set_gripper_value(wp["gripper_percent"], speed=60):
                print(f"❌ 第 {i+1} 个点位夹爪设置失败，停止回放。")
                return False
            if not self.driver.send_angles(wp["joint_angles_deg"], speed=15):
                print(f"❌ 第 {i+1} 个点位移动失败，停止回放。")
                return False
            time.sleep(0.6)

        print("🎉 myCobot Pro 450 轨迹复现回放圆满完成！\n")
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="myCobot Pro 450 Drag-Teach Recorder")
    parser.add_argument("--ip", type=str, default=DEFAULT_ROBOT_IP, help=f"Robot IP (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_ROBOT_PORT, help=f"Robot Port (default: {DEFAULT_ROBOT_PORT})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = DragTeachRecorderApp(ip=args.ip, port=args.port)
    app.run()
