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
import math
import time
import threading
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any

from json_store import read_json, update_json
from camera_device import open_robot_camera

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "calibration_config.json")
SCHEMA_VERSION = 2
MAX_VALID_RMSE_MM = 15.0
MAX_VALID_COND = 1e8

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

def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_homography_and_points(
    img_pts: Optional[np.ndarray],
    rob_pts: Optional[np.ndarray],
    H: Optional[np.ndarray],
    max_rmse_mm: float = MAX_VALID_RMSE_MM,
    max_cond: float = MAX_VALID_COND,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """校验标定点集与单应性矩阵是否非退化、有限且误差在可接受范围。"""
    metrics: Dict[str, Any] = {"rmse_mm": None, "max_error_mm": None, "cond": None}
    if H is None:
        return False, "单应性矩阵为空", metrics
    H_arr = np.asarray(H, dtype=np.float64)
    if H_arr.shape != (3, 3) or not np.all(np.isfinite(H_arr)):
        return False, "单应性矩阵维度不为 3x3 或包含非有限值", metrics
    det = float(np.linalg.det(H_arr))
    if abs(det) < 1e-12:
        return False, f"单应性矩阵接近奇异 (det={det:.2e})", metrics
    cond = float(np.linalg.cond(H_arr))
    metrics["cond"] = cond
    if not math.isfinite(cond) or cond > max_cond:
        return False, f"单应性矩阵条件数过大 ({cond:.2e} > {max_cond:.2e})", metrics

    if img_pts is not None and rob_pts is not None:
        if len(img_pts) < 4 or len(rob_pts) < 4 or len(img_pts) != len(rob_pts):
            return False, "标定点数量不足 4 个或图像/机械臂点数不一致", metrics
        if not np.all(np.isfinite(img_pts)) or not np.all(np.isfinite(rob_pts)):
            return False, "标定点包含非有限坐标", metrics
        # 共线 / 零面积退化检测
        centered_img = img_pts - np.mean(img_pts, axis=0)
        centered_rob = rob_pts - np.mean(rob_pts, axis=0)
        s_img = np.linalg.svd(centered_img, compute_uv=False)
        s_rob = np.linalg.svd(centered_rob, compute_uv=False)
        if len(s_img) < 2 or s_img[1] < 1.0 or len(s_rob) < 2 or s_rob[1] < 1.0:
            return False, "标定点近似共线或局部重合，无法构成有效平面映射", metrics

        pts_in = np.asarray(img_pts, dtype=np.float32).reshape(1, -1, 2)
        pred = cv2.perspectiveTransform(pts_in, H_arr)[0]
        if not np.all(np.isfinite(pred)):
            return False, "重投影结果包含非有限数值", metrics
        errs = np.linalg.norm(pred.astype(np.float64) - np.asarray(rob_pts, dtype=np.float64), axis=1)
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        max_err = float(np.max(errs))
        metrics["rmse_mm"] = round(rmse, 4)
        metrics["max_error_mm"] = round(max_err, 4)
        if rmse > max_rmse_mm:
            return False, f"重投影 RMSE 过大 ({rmse:.2f}mm > {max_rmse_mm:.1f}mm)", metrics
    return True, None, metrics


def save_calibration(
    updates: Dict[str, Any],
    config_path: str = CONFIG_FILE,
    preserve_fields: Tuple[str, ...] = (
        "manual_offset_mm",
        "image_size_px",
        "camera_matrix",
        "dist_coeffs",
        "board_grid",
    ),
) -> Dict[str, Any]:
    """原子更新并持久化手眼标定配置文件，保留未知扩展字段与手工微调偏置。"""
    def _mutate(existing: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(existing) if isinstance(existing, dict) else {}
        for k, v in updates.items():
            merged[k] = v
        for pk in preserve_fields:
            if pk not in updates and isinstance(existing, dict) and pk in existing:
                merged[pk] = existing[pk]
        merged["schema_version"] = SCHEMA_VERSION
        merged.setdefault("calibration_time", time.strftime("%Y-%m-%d %H:%M:%S"))
        merged["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return merged

    return update_json(config_path, _mutate, default={}, allow_corrupt_overwrite=False)


def load_calibration(config_path: str = CONFIG_FILE) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """通过带锁与原子读工具加载标定 JSON。"""
    return read_json(config_path)


class HandEyeMapper:
    """手眼单应性坐标转换矩阵计算器 (支持已保存矩阵复用、校验证据链、版本失效通知与原子热重载)"""

    def __init__(self, config_path: str = CONFIG_FILE):
        self.config_path = config_path
        self._lock = threading.RLock()
        self.version = 0
        self.available = False
        self.unavailable_reason: Optional[str] = "未初始化"
        self.source = "uninitialized"
        self.schema_version = SCHEMA_VERSION
        self.matrix = None
        self.image_points = None
        self.robot_points = None
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.grid_points = None
        self.boundary_polygon = None
        self.table_z = DEFAULT_CALIB_DATA["table_z_mm"]
        self.hover_z = DEFAULT_CALIB_DATA["hover_clearance_mm"]
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.reproj_rmse_mm: Optional[float] = None
        self.reproj_max_mm: Optional[float] = None
        self.condition_number: Optional[float] = None
        self._last_mtime = None
        self.load_config()

    @staticmethod
    def _parse_manual_offset(raw) -> Tuple[float, float, float]:
        """解析标定配置中保存的手动微调偏置 [ΔX, ΔY, ΔZ]"""
        try:
            if isinstance(raw, dict):
                vals = [raw.get("x", 0.0), raw.get("y", 0.0), raw.get("z", 0.0)]
            elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
                vals = [raw[0], raw[1], raw[2]]
            else:
                return (0.0, 0.0, 0.0)
            if all(_is_finite(v) for v in vals):
                return (round(float(vals[0]), 1), round(float(vals[1]), 1), round(float(vals[2]), 1))
        except (TypeError, ValueError):
            pass
        return (0.0, 0.0, 0.0)

    def _mark_unavailable(self, reason: str, bump_version: bool = True, mtime: Optional[float] = None):
        with self._lock:
            self.available = False
            self.unavailable_reason = reason
            self.matrix = None
            if mtime is not None:
                self._last_mtime = mtime
            if bump_version:
                self.version += 1

    def load_config(self) -> bool:
        """原子解析并校验标定文件；文件损坏或标定无效时标记不可用而非静默回退默认值。"""
        with self._lock:
            file_exists = os.path.exists(self.config_path)
            if file_exists:
                try:
                    mtime = os.path.getmtime(self.config_path)
                except OSError:
                    mtime = None
                data, err = read_json(self.config_path)
                if err is not None or not isinstance(data, dict):
                    msg = f"标定文件读取或 JSON 解析失败: {err or '格式不是对象'}"
                    print(f"⚠️ {msg}")
                    self._mark_unavailable(msg, bump_version=True, mtime=mtime)
                    return False
                source = "file"
            else:
                mtime = None
                data = dict(DEFAULT_CALIB_DATA)
                source = "default_fallback"

            try:
                raw_img = data.get("image_points")
                raw_rob = data.get("robot_points")
                has_H = isinstance(data.get("homography_matrix"), (list, tuple))
                img_pts = np.array(raw_img, dtype=np.float32) if raw_img is not None else None
                rob_pts = np.array(raw_rob, dtype=np.float32) if raw_rob is not None else None

                if img_pts is not None and (img_pts.ndim != 2 or img_pts.shape[1] != 2):
                    raise ValueError("image_points 维度无效")
                if rob_pts is not None and (rob_pts.ndim != 2 or rob_pts.shape[1] != 2):
                    raise ValueError("robot_points 维度无效")

                if has_H:
                    H = np.array(data["homography_matrix"], dtype=np.float64)
                    H_source = "saved_matrix"
                else:
                    if img_pts is None or rob_pts is None or len(img_pts) < 4:
                        raise ValueError("缺少有效的 homography_matrix 与足够的标定点对 (>=4)")
                    if len(img_pts) >= 8:
                        H, _ = cv2.findHomography(img_pts, rob_pts, cv2.RANSAC, 3.0)
                        if H is None:
                            H, _ = cv2.findHomography(img_pts, rob_pts)
                    else:
                        H, _ = cv2.findHomography(img_pts, rob_pts)
                    H_source = "fitted_from_points"

                ok, reason, metrics = validate_homography_and_points(img_pts, rob_pts, H)
                if not ok:
                    raise ValueError(reason or "单应性矩阵校验未通过")

                table_z = float(data.get("table_z_mm", DEFAULT_CALIB_DATA["table_z_mm"]))
                hover_z = float(data.get("hover_clearance_mm", table_z + 110.0))
                if not (_is_finite(table_z) and _is_finite(hover_z)):
                    raise ValueError("table_z_mm / hover_clearance_mm 非有限数字")

                ox, oy, oz = self._parse_manual_offset(data.get("manual_offset_mm"))

                boundary = data.get("boundary_polygon")
                boundary_poly = np.array(boundary, dtype=np.float32) if boundary and len(boundary) >= 3 else None

                grid_pts = None
                grid = data.get("board_grid")
                if grid and img_pts is not None and len(img_pts) == 4:
                    cols, rows = int(grid["columns"]), int(grid["rows"])
                    if cols >= 2 and rows >= 2:
                        corners = np.array(
                            [[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]],
                            dtype=np.float32,
                        )
                        grid_matrix = cv2.getPerspectiveTransform(corners, img_pts)
                        points = np.array([[[x, y] for y in range(rows) for x in range(cols)]], dtype=np.float32)
                        grid_pts = cv2.perspectiveTransform(points, grid_matrix)[0]

                cam_mat = None
                dist_coeffs = None
                if data.get("camera_matrix") is not None and data.get("dist_coeffs") is not None:
                    cm = np.array(data["camera_matrix"], dtype=np.float64)
                    dc = np.array(data["dist_coeffs"], dtype=np.float64)
                    if cm.shape == (3, 3) and np.all(np.isfinite(cm)) and np.all(np.isfinite(dc)):
                        cam_mat = cm
                        dist_coeffs = dc

                # 全部校验通过后再原子切换状态
                self.schema_version = int(data.get("schema_version", 1))
                self.source = f"{source}:{H_source}"
                self.matrix = H
                self.image_points = img_pts
                self.robot_points = rob_pts
                self.image_size = data.get("image_size_px")
                self.camera_matrix = cam_mat
                self.dist_coeffs = dist_coeffs
                self.grid_points = grid_pts
                self.boundary_polygon = boundary_poly
                self.table_z = table_z
                self.hover_z = hover_z
                self.offset_x, self.offset_y, self.offset_z = ox, oy, oz
                self.reproj_rmse_mm = metrics["rmse_mm"]
                self.reproj_max_mm = metrics["max_error_mm"]
                self.condition_number = metrics["cond"]
                self.available = True
                self.unavailable_reason = None
                self._last_mtime = mtime
                self.version += 1
                print("✅ 手眼标定矩阵加载完成。")
                return True
            except Exception as e:
                msg = f"标定校验失败: {e}"
                print(f"⚠️ {msg}")
                self._mark_unavailable(msg, bump_version=True, mtime=mtime)
                return False

    def check_reload(self) -> bool:
        """如果磁盘上的标定配置文件被更新，自动重新加载最新参数；返回是否发生重载。"""
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    mtime = os.path.getmtime(self.config_path)
                    if getattr(self, "_last_mtime", None) is None:
                        self._last_mtime = mtime
                    elif mtime != self._last_mtime:
                        self.load_config()
                        return True
                except OSError:
                    pass
            return False

    def check_frame_size(self, width: int, height: int) -> Tuple[bool, Optional[str]]:
        """检查当前相机分辨率/纵横比是否与标定配置一致。"""
        if not (isinstance(width, (int, float)) and isinstance(height, (int, float)) and width > 0 and height > 0):
            return False, f"无效帧分辨率: {width}x{height}"
        if self.image_size and isinstance(self.image_size, (list, tuple)) and len(self.image_size) >= 2:
            cw, ch = float(self.image_size[0]), float(self.image_size[1])
            if cw > 0 and ch > 0:
                cal_ratio = cw / ch
                cur_ratio = float(width) / float(height)
                if abs(cur_ratio - cal_ratio) / cal_ratio > 0.08:
                    return False, f"画面纵横比不一致 (标定 {cw:.0f}x{ch:.0f}, 当前 {int(width)}x{int(height)})"
        return True, None

    def status_dict(self) -> Dict[str, Any]:
        """返回标定模块健康状态摘要。"""
        with self._lock:
            return {
                "available": self.available,
                "reason": self.unavailable_reason,
                "version": self.version,
                "schema_version": self.schema_version,
                "source": self.source,
                "rmse_mm": self.reproj_rmse_mm,
                "max_error_mm": self.reproj_max_mm,
                "table_z_mm": self.table_z,
                "image_size_px": self.image_size,
                "has_intrinsics": self.camera_matrix is not None,
            }

    def get_offset(self) -> Tuple[float, float, float]:
        """获取当前手眼微调偏置 (ΔX, ΔY, ΔZ mm)"""
        with self._lock:
            self.check_reload()
            return (
                round(float(self.offset_x), 1),
                round(float(self.offset_y), 1),
                round(float(self.offset_z), 1),
            )

    def set_offset(self, dx: float, dy: float, dz: float, persist: bool = True) -> Tuple[float, float, float]:
        """设置并可选原子持久化手眼微调偏置 (ΔX, ΔY, ΔZ mm)"""
        if not (_is_finite(dx) and _is_finite(dy) and _is_finite(dz)):
            raise ValueError("手眼微调偏置必须为有限数值")
        with self._lock:
            self.offset_x = round(float(dx), 1)
            self.offset_y = round(float(dy), 1)
            self.offset_z = round(float(dz), 1)

            if persist:
                try:
                    save_calibration(
                        {"manual_offset_mm": [self.offset_x, self.offset_y, self.offset_z]},
                        config_path=self.config_path,
                    )
                    if os.path.exists(self.config_path):
                        self._last_mtime = os.path.getmtime(self.config_path)
                except Exception as e:
                    print(f"⚠️ 保存手眼微调偏置到标定文件失败: {e}")

            return (self.offset_x, self.offset_y, self.offset_z)

    def contains_pixel(self, norm_x: float, norm_y: float) -> bool:
        """判断归一化画面点是否位于标定区域内。"""
        if not (_is_finite(norm_x) and _is_finite(norm_y)):
            return False
        with self._lock:
            self.check_reload()
            if not self.available or self.image_points is None or len(self.image_points) < 3:
                return False
            if self.boundary_polygon is not None and len(self.boundary_polygon) >= 3:
                distance = cv2.pointPolygonTest(self.boundary_polygon, (float(norm_x), float(norm_y)), True)
                return distance >= -25.0
            if len(self.image_points) > 4:
                hull = cv2.convexHull(self.image_points.astype(np.float32))
                distance = cv2.pointPolygonTest(hull, (float(norm_x), float(norm_y)), True)
                return distance >= -25.0
            distance = cv2.pointPolygonTest(self.image_points, (float(norm_x), float(norm_y)), True)
            return distance >= -25.0

    def save_config(
        self,
        img_pts: List[List[float]],
        rob_pts: List[List[float]],
        table_z: float,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """校验并原子保存标定参数（保留原有未知字段与微调偏置）。"""
        img_arr = np.asarray(img_pts, dtype=np.float32)
        rob_arr = np.asarray(rob_pts, dtype=np.float32)
        if len(img_arr) < 4 or len(img_arr) != len(rob_arr):
            raise ValueError("标定点数量不足 4 个或图像/机械臂点数不一致")
        H = None
        if len(img_arr) >= 8:
            H, _ = cv2.findHomography(img_arr, rob_arr, cv2.RANSAC, 3.0)
        if H is None:
            H, _ = cv2.findHomography(img_arr, rob_arr)
        ok, reason, metrics = validate_homography_and_points(img_arr, rob_arr, H)
        if not ok:
            raise ValueError(f"标定数据校验未通过，未保存: {reason}")

        payload: Dict[str, Any] = {
            "image_points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in img_arr],
            "robot_points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in rob_arr],
            "table_z_mm": round(float(table_z), 2),
            "hover_clearance_mm": round(float(table_z) + 110.0, 2),
            "homography_matrix": H.tolist(),
            "rmse_mm": metrics["rmse_mm"],
            "max_error_mm": metrics["max_error_mm"],
        }
        if extra:
            payload.update(extra)
        save_calibration(payload, config_path=self.config_path)
        self.load_config()
        print(f"💾 新标定参数已成功保存至: {self.config_path}")

    def _undistort_norm(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """当标定文件中存在相机内参与畸变系数时，对归一化坐标执行去畸变。"""
        if (
            self.camera_matrix is None
            or self.dist_coeffs is None
            or not self.image_size
            or len(self.image_size) < 2
        ):
            return float(norm_x), float(norm_y)
        w_px, h_px = float(self.image_size[0]), float(self.image_size[1])
        if w_px <= 0 or h_px <= 0:
            return float(norm_x), float(norm_y)
        px = np.array([[[norm_x * w_px / 1000.0, norm_y * h_px / 1000.0]]], dtype=np.float64)
        und = cv2.undistortPoints(px, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return float(und[0][0][0]) * 1000.0 / w_px, float(und[0][0][1]) * 1000.0 / h_px

    def pixel_to_robot(self, norm_x: float, norm_y: float, apply_offset: bool = False) -> Tuple[float, float, float]:
        """
        将归一化像素坐标 (0~1000) 转换为机械臂笛卡尔物理坐标 (X, Y, Z mm)。
        标定不可用或结果异常时显式抛出异常，不静默返回错误坐标或硬钳位掩盖超界。
        """
        if not (_is_finite(norm_x) and _is_finite(norm_y)):
            raise ValueError(f"像素坐标必须为有限数字，当前为 ({norm_x!r}, {norm_y!r})")
        with self._lock:
            self.check_reload()
            if not self.available or self.matrix is None:
                raise RuntimeError(self.unavailable_reason or "手眼标定矩阵不可用")
            ux, uy = self._undistort_norm(float(norm_x), float(norm_y))
            pt = np.array([[[ux, uy]]], dtype=np.float32)
            trans_pt = cv2.perspectiveTransform(pt, self.matrix)
            x_mm = float(trans_pt[0][0][0])
            y_mm = float(trans_pt[0][0][1])
            z_mm = float(self.table_z)

            if apply_offset:
                x_mm += self.offset_x
                y_mm += self.offset_y
                z_mm += self.offset_z

        if not (_is_finite(x_mm) and _is_finite(y_mm) and _is_finite(z_mm)):
            raise ValueError("坐标变换产生非有限物理坐标")
        if not (-466.0 <= x_mm <= 466.0 and -466.0 <= y_mm <= 466.0):
            raise ValueError(f"映射物理坐标越过机械臂包络: ({x_mm:.1f}, {y_mm:.1f}) mm")
        if math.hypot(x_mm, y_mm) > 475.0:
            raise ValueError(f"映射半径超过 Pro 450 最大臂展: ({x_mm:.1f}, {y_mm:.1f}) mm")

        return round(x_mm, 1), round(y_mm, 1), round(z_mm, 1)

    def bbox_to_robot_center(self, bbox: List[int], apply_offset: bool = False) -> Tuple[float, float, float]:
        """输入 [ymin, xmin, ymax, xmax] 格式检测框，计算几何中心并映射为物理坐标。"""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox 必须包含 4 个数值 [ymin, xmin, ymax, xmax]")
        ymin, xmin, ymax, xmax = bbox
        if not all(_is_finite(v) for v in (ymin, xmin, ymax, xmax)):
            raise ValueError("bbox 必须全部为有限数值")
        if ymax < ymin or xmax < xmin:
            raise ValueError(f"bbox 上下界顺序错误: {bbox!r}")
        center_x = (float(xmin) + float(xmax)) / 2.0
        center_y = (float(ymin) + float(ymax)) / 2.0
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
