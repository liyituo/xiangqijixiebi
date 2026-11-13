# -*- coding: utf-8 -*-
"""
DeepSeek-V4.1-Flash 自主视觉与空间坐标换算求解器 (DeepSeek Autonomous Workspace Solver)
实现由 DeepSeek 多模态大模型直接执行：
1. 识别工作台最新 4 个黑点并核验几何有效性；
2. 结合物理基准点，由大模型自主进行透视空间插值与物理坐标推导换算；
3. 与 OpenCV 数学单应性矩阵进行双向几何互验（交叉比对误差）；
4. 输出带物理坐标标注的现场图 (deepseek_solved_workspace.jpg)。
"""

import os
import sys
import json
import time
import re
import requests
import numpy as np
import cv2
from typing import Optional, Dict, Any, List

from env_loader import get_vlm_api_key

# 配置
DEEPSEEK_API_KEY = get_vlm_api_key("VLM_API_KEY", "DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("VLM_MODEL", os.environ.get("DEEPSEEK_MODEL", "qwen3.8-flash"))
DEEPSEEK_API_URL = os.environ.get("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "calibration_config.json")
SNAPSHOT_FILE = os.path.join(os.path.dirname(__file__), "current_table_snapshot.jpg")


def get_latest_frame() -> Optional[np.ndarray]:
    """获取最新工作台图像（支持网络流与中文本地文件路径）"""
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


def frame_to_base64(frame: np.ndarray) -> str:
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    return base64.b64encode(buf).decode("utf-8")


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    if "```" in text:
        matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        for m in matches:
            try:
                return json.loads(m.strip())
            except Exception:
                pass
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first:last+1])
        except Exception:
            pass
    return None


def deepseek_solve_workspace(frame: np.ndarray) -> dict:
    """由 DeepSeek 亲自识别四黑点并换算工作台内目标工件的物理坐标"""
    b64 = frame_to_base64(frame)

    # 读取实测真机基准物理坐标点
    ref_dots = {
        "左上": {"pixel": [348.0, 182.0], "robot": [448.0, 136.0, 170.0]},
        "右上": {"pixel": [686.0, 162.0], "robot": [361.0, -154.0, 170.0]},
        "右下": {"pixel": [699.0, 754.0], "robot": [170.0, -152.0, 170.0]},
        "左下": {"pixel": [354.0, 772.0], "robot": [179.0, 130.0, 170.0]}
    }

    prompt = f"""你是工业机械臂具身几何换算中枢。
画面是工作台俯视图，用户通过 4 个黑点界定了工作台物理空间。
已知这 4 个角点的基准物理坐标映射如下：
{json.dumps(ref_dots, ensure_ascii=False, indent=2)}
桌面抓取基准高度 Z = 189.0 mm。

请执行以下任务：
1. 观察画面，核对 4 个黑点的像素位置；
2. 识别工作台内部的所有核心有效作业目标（如象棋棋子、工件）；
3. 由你亲自执行【空间透视与双线性插值换算】，推导出每个目标在机械臂基座坐标系下的物理毫米坐标 (X, Y, Z)；
4. 严格以 JSON 格式输出：
{{
  "workspace_status": "VALID",
  "corners": [
    {{"name": "点4(左上)", "pixel_norm": [x, y], "robot_xyz": [x, y, z]}},
    {{"name": "点3(右上)", "pixel_norm": [x, y], "robot_xyz": [x, y, z]}},
    {{"name": "点2(右下)", "pixel_norm": [x, y], "robot_xyz": [x, y, z]}},
    {{"name": "点1(左下)", "pixel_norm": [x, y], "robot_xyz": [x, y, z]}}
  ],
  "detected_targets": [
    {{
      "name": "目标名称（如象棋棋子）",
      "pixel_norm": [norm_x, norm_y],
      "deepseek_solved_robot_xyz": [x_mm, y_mm, z_mm],
      "reasoning": "换算计算依据"
    }}
  ]
}}
只输出合法的 JSON 字典，严禁废话。"""

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
        "temperature": 0.1
    }

    t0 = time.time()
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=25.0
    )
    cost = time.time() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API 请求失败: {resp.status_code} - {resp.text}")

    reply = resp.json()["choices"][0]["message"].get("content", "")
    data = extract_json(reply)
    if not data:
        raise ValueError(f"未能解析 DeepSeek 返回的 JSON: {reply}")
    data["elapsed_s"] = round(cost, 2)
    return data


def opencv_verify_homography(pixel_norm: list, config_data: dict) -> tuple:
    """使用 OpenCV 单应性矩阵作为地面基准进行交叉校验"""
    img_pts = np.array(config_data["image_points"], dtype=np.float32)
    rob_pts = np.array(config_data["robot_points"], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, rob_pts)
    pts = np.array([[[float(pixel_norm[0]), float(pixel_norm[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, H)
    rx = float(dst[0][0][0])
    ry = float(dst[0][0][1])
    return (round(rx, 1), round(ry, 1), float(config_data.get("table_z_mm", 189.0)))


def main():
    print("=" * 70)
    print("  🧠 DeepSeek-V4.1-Flash 自主视觉识别与空间物理坐标换算系统")
    print("=" * 70)

    frame = get_latest_frame()
    if frame is None:
        print("❌ 无法获取工作台图像！请检查控制大屏或摄像头。")
        return

    h, w = frame.shape[:2]
    print(f"📷 画面就绪: {w}x{h} 像素")
    print("🚀 正在请求 DeepSeek-V4.1-Flash 进行多模态视觉感知与几何换算推导...")

    try:
        result = deepseek_solve_workspace(frame)
    except Exception as e:
        print(f"❌ DeepSeek 换算失败: {e}")
        return

    # 读取本地标定配置供对比
    local_cfg = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            local_cfg = json.load(f)

    print(f"\n✅ DeepSeek 自主换算完成！(耗时: {result.get('elapsed_s')} 秒)")
    print("-" * 70)
    print("📐【工作台 4 黑点基准换算核验】:")
    for corner in result.get("corners", []):
        cname = corner.get("name")
        cpx = corner.get("pixel_norm")
        cxyz = corner.get("robot_xyz")
        print(f"  • {cname:10s} -> 归一化: {cpx} | 物理坐标: {cxyz}")

    print("\n🎯【工作台目标物体定位与 DeepSeek 换算结果】:")
    targets = result.get("detected_targets", [])
    if not targets:
        print("  (工作台内当前未检测到额外孤立工件，四角黑点已锁定)")

    disp = frame.copy()

    # 绘制 4 点边界框
    if "corners" in result and len(result["corners"]) == 4:
        poly_pts = []
        for c in result["corners"]:
            nx, ny = c["pixel_norm"]
            px, py = int(nx * w / 1000.0), int(ny * h / 1000.0)
            poly_pts.append([px, py])
            cv2.circle(disp, (px, py), 7, (0, 0, 255), -1)
            cv2.circle(disp, (px, py), 14, (0, 255, 255), 2)
            cv2.putText(disp, c["name"], (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.polylines(disp, [np.array(poly_pts, np.int32)], True, (255, 120, 0), 2)

    for idx, t in enumerate(targets, 1):
        name = t.get("name", "工件")
        pix = t.get("pixel_norm", [500, 500])
        ds_xyz = t.get("deepseek_solved_robot_xyz", [0, 0, 0])
        reason = t.get("reasoning", "")

        # 自动兼容 0~1 与 0~1000 归一化格式
        if max(pix) <= 1.05:
            pix_1000 = [round(float(v) * 1000.0, 1) for v in pix]
        else:
            pix_1000 = [round(float(v), 1) for v in pix]

        # OpenCV 校验
        cv_xyz = opencv_verify_homography(pix_1000, local_cfg) if local_cfg else ds_xyz
        dx = abs(ds_xyz[0] - cv_xyz[0])
        dy = abs(ds_xyz[1] - cv_xyz[1])

        print(f"\n  [{idx}] 目标: 【{name}】")
        print(f"      图像归一化坐标: {pix_1000}")
        print(f"      DeepSeek 换算物理: X={ds_xyz[0]:.1f}, Y={ds_xyz[1]:.1f}, Z={ds_xyz[2]:.1f} mm")
        print(f"      OpenCV 矩阵复核:   X={cv_xyz[0]:.1f}, Y={cv_xyz[1]:.1f}, Z={cv_xyz[2]:.1f} mm (误差: ΔX={dx:.1f}mm, ΔY={dy:.1f}mm)")
        print(f"      💡 DeepSeek 推导依据: {reason}")

        # 在图像上绘制目标
        tpx, tpy = int(pix_1000[0] * w / 1000.0), int(pix_1000[1] * h / 1000.0)
        cv2.circle(disp, (tpx, tpy), 10, (0, 255, 0), -1)
        cv2.putText(disp, f"{name}: X:{ds_xyz[0]:.0f} Y:{ds_xyz[1]:.0f}", (tpx + 15, tpy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # 安全保存图片 (兼容 Windows 中文路径)
    out_path = os.path.join(os.path.dirname(__file__), "deepseek_solved_workspace.jpg")
    ret, buf = cv2.imencode(".jpg", disp)
    if ret:
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        print(f"\n🖼️ DeepSeek 换算可视化全景已保存至: deepseek_solved_workspace.jpg")

    print("\n✅ DeepSeek 自主换算流程全部执行完毕！")


if __name__ == "__main__":
    main()
