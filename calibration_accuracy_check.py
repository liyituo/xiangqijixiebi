# -*- coding: utf-8 -*-
"""
手眼标定独立精度核验与交叉验证工具 (Calibration Accuracy & LOOCV Checker)
提供留一法交叉验证 (LOOCV)、重投影误差分布、工作区边界检查与 CLI 报告。
"""

import os
import sys
import math
import argparse
from typing import Dict, Any, Optional, List
import numpy as np
import cv2

from vision_calibration import (
    CONFIG_FILE,
    HandEyeMapper,
    load_calibration,
    validate_homography_and_points,
)


def evaluate_calibration(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """对指定标定配置文件执行独立重投影误差与留一法交叉验证评估。"""
    data, err = load_calibration(config_path)
    if err is not None or not isinstance(data, dict):
        return {
            "ok": False,
            "error": err or "标定文件为空或格式错误",
            "config_path": config_path,
        }

    mapper = HandEyeMapper(config_path=config_path)
    if not mapper.available or mapper.matrix is None:
        return {
            "ok": False,
            "error": mapper.unavailable_reason or "单应性矩阵不可用",
            "config_path": config_path,
        }

    img_pts = mapper.image_points
    rob_pts = mapper.robot_points
    if img_pts is None or rob_pts is None or len(img_pts) < 4:
        return {
            "ok": False,
            "error": "标定点不足 4 组，无法执行完整精度评估",
            "config_path": config_path,
        }

    ok, reason, metrics = validate_homography_and_points(img_pts, rob_pts, mapper.matrix)

    # 留一法交叉验证 (仅当点数 >= 5 时可执行 4+ 点留一预测)
    loocv_errors: List[float] = []
    if len(img_pts) >= 5:
        for i in range(len(img_pts)):
            mask = np.ones(len(img_pts), dtype=bool)
            mask[i] = False
            sub_img = img_pts[mask]
            sub_rob = rob_pts[mask]
            if len(sub_img) >= 8:
                H_sub, _ = cv2.findHomography(sub_img, sub_rob, cv2.RANSAC, 3.0)
                if H_sub is None:
                    H_sub, _ = cv2.findHomography(sub_img, sub_rob)
            else:
                H_sub, _ = cv2.findHomography(sub_img, sub_rob)
            if H_sub is None or not np.all(np.isfinite(H_sub)):
                continue
            pt = np.array([[img_pts[i]]], dtype=np.float32)
            pred = cv2.perspectiveTransform(pt, H_sub)[0][0]
            if np.all(np.isfinite(pred)):
                err_mm = float(math.hypot(pred[0] - rob_pts[i][0], pred[1] - rob_pts[i][1]))
                loocv_errors.append(err_mm)

    loocv_rmse = float(np.sqrt(np.mean(np.square(loocv_errors)))) if loocv_errors else None
    loocv_max = float(np.max(loocv_errors)) if loocv_errors else None

    return {
        "ok": bool(ok),
        "error": reason,
        "config_path": config_path,
        "calibration_type": data.get("calibration_type", "unknown"),
        "schema_version": data.get("schema_version", 1),
        "num_points": int(len(img_pts)),
        "table_z_mm": float(mapper.table_z),
        "manual_offset_mm": list(mapper.get_offset()),
        "condition_number": metrics["cond"],
        "reproj_rmse_mm": metrics["rmse_mm"],
        "reproj_max_mm": metrics["max_error_mm"],
        "loocv_count": len(loocv_errors),
        "loocv_rmse_mm": round(loocv_rmse, 4) if loocv_rmse is not None else None,
        "loocv_max_mm": round(loocv_max, 4) if loocv_max is not None else None,
        "has_intrinsics": mapper.camera_matrix is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration Independent Accuracy Evaluator")
    parser.add_argument("--config", default=CONFIG_FILE, help="Path to calibration_config.json")
    args = parser.parse_args()

    res = evaluate_calibration(args.config)
    if not res.get("ok"):
        print(f"❌ 标定核验失败: {res.get('error')}")
        sys.exit(1)

    print("=" * 64)
    print(f"📊 手眼标定独立评估报告: {res['config_path']}")
    print(f"   • 类型 / Schema: {res['calibration_type']} (v{res['schema_version']})")
    print(f"   • 采样点数: {res['num_points']} | 基准高度 Z: {res['table_z_mm']:.2f} mm")
    print(f"   • 矩阵条件数: {res['condition_number']:.2f} | 相机内参去畸变: {'已启用' if res['has_intrinsics'] else '未配置'}")
    print(f"   • 拟合重投影 RMSE: {res['reproj_rmse_mm']} mm | Max: {res['reproj_max_mm']} mm")
    if res["loocv_rmse_mm"] is not None:
        print(f"   • 留一交叉验证 ({res['loocv_count']}点) RMSE: {res['loocv_rmse_mm']} mm | Max: {res['loocv_max_mm']} mm")
    print("=" * 64)


if __name__ == "__main__":
    main()
