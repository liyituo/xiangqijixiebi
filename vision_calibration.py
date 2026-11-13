# -*- coding: utf-8 -*-
"""
手眼标定与坐标转换模块 (Hand-Eye Calibration & Coordinate Mapper)
用于将 Qwen2.5-VL 视觉大模型输出的图像像素坐标 (0~1000 归一化坐标)，
精确转换到大象机器人工作台的物理毫米坐标 (X, Y, Z)。
"""

import os
import sys
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import numpy as np
import cv2
from typing import Tuple, List, Optional
from camera_device import open_robot_camera

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "calibration_config.json")

# 默认标定参考点 (严格基于 P1~P5 实测关节与笛卡尔示教航点矩阵):
# P1(远端左侧 X:296 Y:207), P3(远端右侧 X:342 Y:-238), P5/P4(近端基座侧)
DEFAULT_CALIB_DATA = {
    "image_points": [
        [150.0, 150.0],  # 靠近 P1 (图像左上)
        [850.0, 150.0],  # 靠近 P3 (图像右上)
        [850.0, 800.0],  # 靠近 P5/急停盒 (图像右下)
        [150.0, 800.0]   # 近端左侧 (图像左下)
    ],
    "robot_points": [
        [296.0, 207.0],   # P1 真实物理示教点 (远端左侧)
        [342.0, -238.0],  # P3 真实物理示教点 (远端右侧)
        [160.0, -238.0],  # 近端右侧 (靠近 P5/急停盒)
        [160.0, 207.0]    # 近端左侧
    ],
    "table_z_mm": 189.0,        # 桌面实际物理抓取基准高度 Z (mm，基于 P1~P5 标定)
    "hover_clearance_mm": 310.0 # 空中过渡/进近高度 Z (mm，基于 NORMAL 待机高度)
}

class HandEyeMapper:
    """手眼单应性坐标转换矩阵计算器"""

    def __init__(self, config_path: str = CONFIG_FILE):
        self.config_path = config_path
        self.matrix = None
        self.image_points = None
        self.image_size = None
        self.grid_points = None
        self.table_z = DEFAULT_CALIB_DATA["table_z_mm"]
        self.hover_z = DEFAULT_CALIB_DATA["hover_clearance_mm"]
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self._last_mtime = None
        self.load_config()

    @staticmethod
    def _parse_manual_offset(raw) -> Tuple[float, float, float]:
        """解析标定配置中保存的手动微调偏置 [ΔX, ΔY, ΔZ]"""
        try:
            if isinstance(raw, dict):
                return (
                    round(float(raw.get("x", 0.0)), 1),
                    round(float(raw.get("y", 0.0)), 1),
                    round(float(raw.get("z", 0.0)), 1),
                )
            if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                return (
                    round(float(raw[0]), 1),
                    round(float(raw[1]), 1),
                    round(float(raw[2]), 1),
                )
        except (TypeError, ValueError):
            pass
        return (0.0, 0.0, 0.0)

    def load_config(self):
        """加载或初始化标定矩阵"""
        data = DEFAULT_CALIB_DATA
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️ 读取标定文件失败，使用默认配置: {e}")

        img_pts = np.array(data["image_points"], dtype=np.float32)
        rob_pts = np.array(data["robot_points"], dtype=np.float32)
        self.image_points = img_pts
        self.image_size = data.get("image_size_px")
        self.grid_points = None
        self.table_z = data.get("table_z_mm", 65.0)
        self.hover_z = data.get("hover_clearance_mm", 150.0)
        self.offset_x, self.offset_y, self.offset_z = self._parse_manual_offset(
            data.get("manual_offset_mm")
        )

        boundary = data.get("boundary_polygon")
        if boundary and len(boundary) >= 3:
            self.boundary_polygon = np.array(boundary, dtype=np.float32)
        else:
            self.boundary_polygon = None

        # 计算 2D 单应性矩阵 (Homography Matrix)
        self.matrix, _ = cv2.findHomography(img_pts, rob_pts)
        grid = data.get("board_grid")
        if grid and len(img_pts) == 4:
            cols, rows = int(grid["columns"]), int(grid["rows"])
            if cols >= 2 and rows >= 2:
                corners = np.array([[0, 0], [cols - 1, 0],
                                    [cols - 1, rows - 1], [0, rows - 1]], dtype=np.float32)
                grid_matrix = cv2.getPerspectiveTransform(corners, img_pts)
                points = np.array([[[x, y] for y in range(rows) for x in range(cols)]], dtype=np.float32)
                self.grid_points = cv2.perspectiveTransform(points, grid_matrix)[0]
        if os.path.exists(self.config_path):
            try:
                self._last_mtime = os.path.getmtime(self.config_path)
            except Exception:
                pass
        print("✅ 手眼标定矩阵加载完成。")

    def check_reload(self) -> bool:
        """如果磁盘上的标定配置文件被更新，自动重新加载最新参数；返回是否发生重载"""
        if os.path.exists(self.config_path):
            try:
                mtime = os.path.getmtime(self.config_path)
                if getattr(self, '_last_mtime', None) is None:
                    self._last_mtime = mtime
                elif mtime != self._last_mtime:
                    self.load_config()
                    self._last_mtime = mtime
                    return True
            except Exception:
                pass
        return False

    def get_offset(self) -> Tuple[float, float, float]:
        """获取当前手眼微调偏置 (ΔX, ΔY, ΔZ mm)"""
        self.check_reload()
        return (
            round(float(self.offset_x), 1),
            round(float(self.offset_y), 1),
            round(float(self.offset_z), 1),
        )

    def set_offset(self, dx: float, dy: float, dz: float, persist: bool = True) -> Tuple[float, float, float]:
        """设置并可选持久化手眼微调偏置 (ΔX, ΔY, ΔZ mm)"""
        self.offset_x = round(float(dx), 1)
        self.offset_y = round(float(dy), 1)
        self.offset_z = round(float(dz), 1)

        if persist and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["manual_offset_mm"] = [self.offset_x, self.offset_y, self.offset_z]
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self._last_mtime = os.path.getmtime(self.config_path)
            except Exception as e:
                print(f"⚠️ 保存手眼微调偏置到标定文件失败: {e}")

        return (self.offset_x, self.offset_y, self.offset_z)

    def contains_pixel(self, norm_x: float, norm_y: float) -> bool:
        """判断归一化画面点是否位于标定区域内。"""
        self.check_reload()
        if self.image_points is None or len(self.image_points) < 3:
            return False
        if hasattr(self, 'boundary_polygon') and self.boundary_polygon is not None and len(self.boundary_polygon) >= 3:
            distance = cv2.pointPolygonTest(self.boundary_polygon, (float(norm_x), float(norm_y)), True)
            return distance >= -25.0
        if len(self.image_points) > 4:
            hull = cv2.convexHull(self.image_points.astype(np.float32))
            distance = cv2.pointPolygonTest(hull, (float(norm_x), float(norm_y)), True)
            return distance >= -25.0
        distance = cv2.pointPolygonTest(self.image_points, (float(norm_x), float(norm_y)), True)
        return distance >= -25.0

    def save_config(self, img_pts: List[List[float]], rob_pts: List[List[float]], table_z: float):
        """保存用户标定好的四点参数"""
        data = {
            "image_points": img_pts,
            "robot_points": rob_pts,
            "table_z_mm": table_z,
            "hover_clearance_mm": table_z + 80.0
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        self.load_config()
        print(f"💾 新标定参数已成功保存至: {self.config_path}")

    def pixel_to_robot(self, norm_x: float, norm_y: float, apply_offset: bool = False) -> Tuple[float, float, float]:
        """
        将 Qwen2.5-VL 输出的归一化像素坐标 (0~1000) 转换为机械臂笛卡尔物理坐标 (X, Y, Z mm)
        当 apply_offset=True 时，叠加已保存的手眼微调偏置 (ΔX, ΔY, ΔZ)。
        """
        self.check_reload()
        pt = np.array([[[norm_x, norm_y]]], dtype=np.float32)
        trans_pt = cv2.perspectiveTransform(pt, self.matrix)
        x_mm = float(trans_pt[0][0][0])
        y_mm = float(trans_pt[0][0][1])
        z_mm = float(self.table_z)

        if apply_offset:
            x_mm += self.offset_x
            y_mm += self.offset_y
            z_mm += self.offset_z

        # 安全边界限位审查 (安全工作圈防飞车硬拦截，适配 Pro 450 全工作台覆盖)
        x_mm = max(50.0, min(500.0, x_mm))
        y_mm = max(-360.0, min(360.0, y_mm))

        return round(x_mm, 1), round(y_mm, 1), round(z_mm, 1)

    def bbox_to_robot_center(self, bbox: List[int], apply_offset: bool = False) -> Tuple[float, float, float]:
        """
        输入 [ymin, xmin, ymax, xmax] 格式检测框，计算几何中心并映射为物理坐标
        """
        ymin, xmin, ymax, xmax = bbox
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        return self.pixel_to_robot(center_x, center_y, apply_offset=apply_offset)


def run_interactive_calibration():
    """交互式四点手眼标定工具"""
    print("=" * 65)
    print("  🎯 机械臂四点手眼快速标定工具 (Click-to-Calibrate)")
    print("=" * 65)
    print("步骤说明:")
    print("1. 界面将打开摄像头画面；")
    print("2. 请依次点击画面中 4 个标定物的位置（建议按：左上、右上、右下、左下 顺时针点击）；")
    print("3. 每点击一次，在控制台输入机械臂移动到该物块上方的真实 (X, Y) 坐标；")
    print("4. 全部完成后自动计算透视单应性矩阵并保存。\n")

    try:
        cap = open_robot_camera()
    except RuntimeError as error:
        print(f"❌ {error}")
        return

    pts = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            h, w = param[:2]
            norm_x = (x / w) * 1000.0
            norm_y = (y / h) * 1000.0
            pts.append((x, y, norm_x, norm_y))
            print(f"📍 捕获第 {len(pts)} 个点: 像素({x}, {y}) -> 归一化({norm_x:.1f}, {norm_y:.1f})")

    cv2.namedWindow("Calibration")
    ret, frame = cap.read()
    if not ret:
        print("读取画面失败。")
        cap.release()
        return

    cv2.setMouseCallback("Calibration", mouse_callback, frame.shape)

    while len(pts) < 4:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        for idx, (px, py, _, _) in enumerate(pts):
            cv2.circle(disp, (px, py), 8, (0, 0, 255), -1)
            cv2.putText(disp, f"P{idx+1}", (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, f"Click 4 points: {len(pts)}/4", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Calibration", disp)
        if cv2.waitKey(10) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("已取消标定。")
            return

    cap.release()
    cv2.destroyAllWindows()

    img_points = [[p[2], p[3]] for p in pts]
    robot_points = []
    print("\n✍️ 请输入 4 个点对应的机械臂真实毫米坐标 (可在大象机器人上位机读取):")
    for i in range(4):
        raw = input(f"第 {i+1} 点物理坐标 (格式如 '180, -120'): ").strip()
        parts = [float(x.strip()) for x in raw.split(",")]
        robot_points.append(parts)

    z_raw = input("请输入抓取桌面 Z 高度 (mm，例如 65): ").strip()
    table_z = float(z_raw) if z_raw else 65.0

    mapper = HandEyeMapper()
    mapper.save_config(img_points, robot_points, table_z)
    print("🎉 标定完成，已生效！")

if __name__ == "__main__":
    mapper = HandEyeMapper()
    # 测试默认映射
    x, y, z = mapper.bbox_to_robot_center([400, 400, 600, 600])
    print(f"测试映射：图像中心 [400,400,600,600] -> 机械臂物理坐标: X={x}mm, Y={y}mm, Z={z}mm")
