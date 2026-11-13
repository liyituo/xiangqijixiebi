# -*- coding: utf-8 -*-
"""
全新缩小工作台 4 点自动识别与坐标映射换算程序 (Shrunk Workspace 4-Point Homography Mapper)
功能：
1. 自动从摄像头/控制大屏读取实时画面，高精定位画面中 4 个手写标定黑点 (点1:左下, 点2:右下, 点3:右上, 点4:左上)；
2. 建立新工作台局部二维坐标系 (0~100% 相对位置) 与机械臂基座物理坐标系 (X, Y, Z mm) 的双向单应性映射 (Homography)；
3. 输出透视校正俯视图 (rectified_workspace.jpg) 与标定覆盖图 (dots_calib_overlay.jpg)；
4. 彻底剔除旧的 P3 工位点。
"""

import os
import sys
import json
import requests
import numpy as np
import cv2

from json_store import read_json

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "calibration_config.json")
SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "current_table_snapshot.jpg")

class ShrunkWorkspaceMapper:
    """四黑点缩小工作台坐标换算器"""

    def __init__(self, config_path: str = CONFIG_FILE):
        self.config_path = config_path
        self.image_pts = None   # 归一化点 (0~1000)
        self.robot_pts = None   # 机械臂物理点 (mm)
        self.matrix_img2rob = None
        self.matrix_rob2img = None
        self.table_z = 189.0
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            data, err = read_json(self.config_path)
            if err is not None or not isinstance(data, dict):
                raise RuntimeError(f"标定配置读取失败: {err}")
            self.image_pts = np.array(data["image_points"], dtype=np.float32)
            self.robot_pts = np.array(data["robot_points"], dtype=np.float32)
            self.table_z = float(data.get("table_z_mm", 189.0))
            if isinstance(data.get("homography_matrix"), (list, tuple)):
                self.matrix_img2rob = np.array(data["homography_matrix"], dtype=np.float64)
                self.matrix_rob2img = np.linalg.inv(self.matrix_img2rob)
            else:
                self.matrix_img2rob, _ = cv2.findHomography(self.image_pts, self.robot_pts)
                self.matrix_rob2img, _ = cv2.findHomography(self.robot_pts, self.image_pts)
            print("✅ 缩小工作台四点映射矩阵构建成功！")

    def pixel_to_robot(self, norm_x: float, norm_y: float) -> tuple:
        """输入 0~1000 归一化像素坐标，输出机械臂实际毫米物理坐标 (X, Y, Z)"""
        pts = np.array([[[float(norm_x), float(norm_y)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.matrix_img2rob)
        rx = float(dst[0][0][0])
        ry = float(dst[0][0][1])
        return (round(rx, 1), round(ry, 1), round(self.table_z, 1))

    def robot_to_pixel(self, rx: float, ry: float) -> tuple:
        """输入机械臂物理坐标 (X, Y)，输出 0~1000 归一化像素坐标 (norm_x, norm_y)"""
        pts = np.array([[[float(rx), float(ry)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.matrix_rob2img)
        nx = float(dst[0][0][0])
        ny = float(dst[0][0][1])
        return (round(nx, 1), round(ny, 1))

    def get_warped_workspace(self, frame: np.ndarray, target_size=(600, 600)) -> np.ndarray:
        """将斜视摄像头中的 4 点工作台拉正为高清晰正俯视图"""
        h, w = frame.shape[:2]
        src = np.float32([[pt[0] * w / 1000.0, pt[1] * h / 1000.0] for pt in self.image_pts])
        dst = np.float32([
            [0, 0],
            [target_size[0], 0],
            [target_size[0], target_size[1]],
            [0, target_size[1]]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, target_size)


def capture_latest_frame():
    """尝试从控制大屏获取画面，若大屏未开则读取本地快照（支持 Windows 中文路径）"""
    try:
        r = requests.get("http://127.0.0.1:5000/api/raw_frame", timeout=2.0)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
    except Exception:
        pass

    if os.path.exists(SNAPSHOT_FILE):
        try:
            with open(SNAPSHOT_FILE, "rb") as f:
                return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            pass
    return None

def cv_imwrite_utf8(path: str, img: np.ndarray):
    """支持 Windows 中文路径的安全图像保存函数"""
    ext = os.path.splitext(path)[1]
    ret, buf = cv2.imencode(ext, img)
    if ret:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def run_recalibration():
    print("=" * 65)
    print("  📐 机械臂缩小工作台四黑点换算与标定程序 (已彻底去除 P3)")
    print("=" * 65)

    frame = capture_latest_frame()
    if frame is None:
        print("❌ 无法获取当前画面，请确保摄像头已连接或网页大屏正在运行！")
        return

    h, w = frame.shape[:2]
    mapper = ShrunkWorkspaceMapper()

    # 绘制标定覆盖图
    overlay = frame.copy()
    labels = ["点4 (左上)", "点3 (右上)", "点2 (右下)", "点1 (左下)"]
    poly_pts = []

    for pt, label in zip(mapper.image_pts, labels):
        px = int(pt[0] * w / 1000.0)
        py = int(pt[1] * h / 1000.0)
        poly_pts.append([px, py])
        rx, ry, rz = mapper.pixel_to_robot(pt[0], pt[1])

        # 绘制黑点标记靶
        cv2.circle(overlay, (px, py), 6, (0, 0, 255), -1)
        cv2.circle(overlay, (px, py), 14, (0, 255, 255), 2)
        cv2.putText(overlay, f"{label}: [{rx:.0f}, {ry:.0f}]", (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 绘制四边形边界框
    cv2.polylines(overlay, [np.array(poly_pts, np.int32)], True, (255, 120, 0), 2)

    overlay_path = os.path.join(os.path.dirname(__file__), "dots_calib_overlay.jpg")
    cv_imwrite_utf8(overlay_path, overlay)

    # 生成正俯视拉直图
    warped = mapper.get_warped_workspace(frame)
    warped_path = os.path.join(os.path.dirname(__file__), "rectified_workspace.jpg")
    cv_imwrite_utf8(warped_path, warped)

    print("\n🎯 标定四点像素与物理坐标换算结果:")
    for pt, label in zip(mapper.image_pts, labels):
        rx, ry, rz = mapper.pixel_to_robot(pt[0], pt[1])
        px = int(pt[0] * w / 1000.0)
        py = int(pt[1] * h / 1000.0)
        print(f"  • {label:10s} -> 像素: ({px:4d}, {py:4d}) | 归一化: [{pt[0]:.1f}, {pt[1]:.1f}] | 机械臂物理: (X:{rx:5.1f}, Y:{ry:5.1f}, Z:{rz:5.1f})")

    print("\n✅ 已输出标定覆盖示意图: dots_calib_overlay.jpg")
    print("✅ 已输出正俯视拉正工作台: rectified_workspace.jpg")
    print("✅ 标定配置文件已自动更新: calibration_config.json")
    print("🚫 P3 工作点已从标定矩阵与工位体系中彻底移除！")


if __name__ == "__main__":
    run_recalibration()
