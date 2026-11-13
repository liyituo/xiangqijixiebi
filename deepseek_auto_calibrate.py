# -*- coding: utf-8 -*-
"""
DeepSeek-V4.1-Flash 全自主视觉手眼标定系统 (DeepSeek Autonomous Hand-Eye Calibration)
完全免人工手动打点：
由 DeepSeek-V4.1-Flash 多模态视觉模型自主发现画面中的 4 个黑点工作台标记，
自动计算单应性矩阵 (Homography)，自动更新 calibration_config.json，并生成可视化核验大图。
"""

import os
import sys
import json
import time
import re
import requests
import numpy as np
import cv2

from env_loader import get_vlm_api_key
from vision_calibration import save_calibration

DEEPSEEK_API_KEY = get_vlm_api_key("DIRECT_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "calibration_config.json")
SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "current_table_snapshot.jpg")

# 4 个黑点的机械臂真实物理示教点 (真机实测坐标与相机视野空间对齐):
# 机械臂基座在画面左侧：
# A点(图像左上, 靠近基座): X:179.0, Y:130.0
# B点(图像右上, 远离基座): X:448.0, Y:136.0
# C点(图像右下, 远离基座): X:361.0, Y:-154.0
# D点(图像左下, 靠近基座): X:170.0, Y:-152.0
PHYSICAL_ROBOT_CORNERS = [
    [179.0, 130.0],   # A点 (图像左上)
    [448.0, 136.0],   # B点 (图像右上)
    [361.0, -154.0],  # C点 (图像右下)
    [170.0, -152.0]   # D点 (图像左下)
]
TABLE_Z = 170.0
HOVER_Z = 280.0


def capture_fresh_frame() -> np.ndarray:
    """优先从控制大屏拉取实时无畸变画面，若未运行则读最新快照"""
    try:
        r = requests.get("http://127.0.0.1:5000/api/raw_frame", timeout=2.0)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                # 顺便更新快照
                with open(SNAPSHOT_FILE, "wb") as f:
                    f.write(r.content)
                return frame
    except Exception:
        pass

    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE, "rb") as f:
            return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)

    # 尝试直连相机
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return frame

    raise RuntimeError("无法捕获工作台画面，请确认摄像头或网页大屏已开启！")


def run_deepseek_calibration():
    print("=" * 68)
    print("  👁️ DeepSeek-V4.1-Flash 全自主视觉手眼标定程序 (已彻底去除 P3)")
    print("=" * 68)

    print("📷 正在抓取当前工作台最新画面...")
    frame = capture_fresh_frame()
    h, w = frame.shape[:2]
    print(f"✅ 获取画面成功: 分辨率 {w} x {h}")

    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    import base64
    b64 = base64.b64encode(buf).decode("utf-8")

    prompt = """你是工业机器人高精视觉标定中枢。
画面是工作台的实时俯视图，用户在桌面上绘制了 4 个黑点标记，界定了缩小后的全新有效作业区域。
请精确定位这 4 个黑点的亚像素几何中心：
1. 点4(左上角黑点，手绘框左上)；
2. 点3(右上角黑点，手绘框右上)；
3. 点2(右下角黑点，手绘框右下)；
4. 点1(左下角黑点，手绘框左下)。

必须以严格的 JSON 格式输出，坐标归一化到 0~1000 范围：
{
  "dots": [
    {"name": "点4(左上)", "point_norm": [x, y]},
    {"name": "点3(右上)", "point_norm": [x, y]},
    {"name": "点2(右下)", "point_norm": [x, y]},
    {"name": "点1(左下)", "point_norm": [x, y]}
  ]
}
只输出合法 JSON，严禁废话。"""

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0.05
    }

    print("🧠 正在请求 DeepSeek-V4.1-Flash 视觉多模态定位 4 个标定黑点...")
    t0 = time.time()
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=25.0
    )
    cost = round(time.time() - t0, 2)

    if resp.status_code != 200:
        print(f"❌ DeepSeek 调用异常: {resp.status_code} - {resp.text}")
        return

    raw_text = resp.json()["choices"][0]["message"].get("content", "")
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        print(f"❌ 未能解析 JSON: {raw_text}")
        return

    data = json.loads(match.group(0))
    dots_list = data.get("dots", [])
    if len(dots_list) != 4:
        print(f"❌ DeepSeek 返回的点数不是 4 个: {dots_list}")
        return

    print(f"✅ DeepSeek 视觉定位成功！(耗时: {cost} 秒)")
    print("-" * 68)

    # 提取归一化像素点并校对格式
    detected_img_pts = []
    labels = ["点4(左上)", "点3(右上)", "点2(右下)", "点1(左下)"]
    for i, d in enumerate(dots_list):
        pt = d.get("point_norm", [0, 0])
        # 兼容 0~1 与 0~1000 范围
        nx = float(pt[0]) * 1000.0 if max(pt) <= 1.05 else float(pt[0])
        ny = float(pt[1]) * 1000.0 if max(pt) <= 1.05 else float(pt[1])
        detected_img_pts.append([round(nx, 1), round(ny, 1)])

    # 计算透视单应性矩阵
    src_pts = np.array(detected_img_pts, dtype=np.float32)
    dst_pts = np.array(PHYSICAL_ROBOT_CORNERS, dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        print("❌ 单应性矩阵计算失败，标定未保存。")
        return

    # 组装标定数据
    calib_payload = {
        "calibration_type": "deepseek_4_point",
        "image_size_px": [int(w), int(h)],
        "image_points": detected_img_pts,
        "robot_points": PHYSICAL_ROBOT_CORNERS,
        "corners_label": labels,
        "table_z_mm": TABLE_Z,
        "hover_clearance_mm": HOVER_Z,
        "homography_matrix": H.tolist(),
        "calibrated_by": "DeepSeek-V4.1-Flash",
        "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # 原子写入 calibration_config.json (保留手动偏置等扩展字段)
    save_calibration(calib_payload, config_path=CONFIG_FILE)
    print(f"💾 已将 DeepSeek 标定参数原子写入: {CONFIG_FILE}")

    print("\n🎯 DeepSeek 标定手眼对齐矩阵 (Homography):")
    for r in H:
        print(f"   [ {r[0]:11.6f}, {r[1]:11.6f}, {r[2]:11.6f} ]")

    print("\n📍 标定四黑点映射对齐明细:")
    overlay = frame.copy()
    poly_pts = []

    for i, (ipt, rpt, lbl) in enumerate(zip(detected_img_pts, PHYSICAL_ROBOT_CORNERS, labels)):
        px = int(ipt[0] * w / 1000.0)
        py = int(ipt[1] * h / 1000.0)
        poly_pts.append([px, py])

        # 测试正向单应性投影校验
        test_pt = np.array([[[ipt[0], ipt[1]]]], dtype=np.float32)
        proj_rob = cv2.perspectiveTransform(test_pt, H)[0][0]
        err_x = abs(proj_rob[0] - rpt[0])
        err_y = abs(proj_rob[1] - rpt[1])

        print(f"  • {lbl:10s} -> 像素: ({px:4d}, {py:4d}) | 归一化: [{ipt[0]:5.1f}, {ipt[1]:5.1f}] -> 机械臂: (X:{rpt[0]:5.1f}, Y:{rpt[1]:5.1f}) [残差: ΔX={err_x:.2f}mm, ΔY={err_y:.2f}mm]")

        # 绘制黑点准星标记
        cv2.circle(overlay, (px, py), 7, (0, 0, 255), -1)
        cv2.circle(overlay, (px, py), 15, (0, 255, 255), 2)
        cv2.putText(overlay, f"{lbl}: [{rpt[0]:.0f}, {rpt[1]:.0f}]", (px + 12, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 绘制四边形边界框
    cv2.polylines(overlay, [np.array(poly_pts, np.int32)], True, (255, 120, 0), 2)

    out_img = os.path.join(os.path.dirname(__file__), "deepseek_calibration_result.jpg")
    ret, buf = cv2.imencode(".jpg", overlay)
    if ret:
        with open(out_img, "wb") as f:
            f.write(buf.tobytes())
    print(f"\n🖼️ 标定核对图已保存: deepseek_calibration_result.jpg")

    # 生成正俯视拉平图
    warp_size = (600, 600)
    warp_src = np.float32([[pt[0] * w / 1000.0, pt[1] * h / 1000.0] for pt in detected_img_pts])
    warp_dst = np.float32([
        [0, 0],
        [warp_size[0], 0],
        [warp_size[0], warp_size[1]],
        [0, warp_size[1]]
    ])
    M_warp = cv2.getPerspectiveTransform(warp_src, warp_dst)
    warped_view = cv2.warpPerspective(frame, M_warp, warp_size)
    warp_out = os.path.join(os.path.dirname(__file__), "deepseek_calibrated_warped.jpg")
    ret, buf = cv2.imencode(".jpg", warped_view)
    if ret:
        with open(warp_out, "wb") as f:
            f.write(buf.tobytes())
    print(f"🖼️ 正俯视校正工作台已保存: deepseek_calibrated_warped.jpg")

    print("\n🎉【DeepSeek 全自主视觉手眼标定全部完成！】")
    print("👉 旧的 P3 工位点已彻底剔除，新的四点工作台矩阵即刻生效。")


if __name__ == "__main__":
    run_deepseek_calibration()
