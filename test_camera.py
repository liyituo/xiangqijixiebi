# -*- coding: utf-8 -*-
"""
摄像头测试与画面捕获脚本
用于检测新插入的摄像头设备，预览实时画面，并抓取用于大模型视觉推理的图像。
"""

import cv2
import os
import time

def test_cameras():
    print("=" * 60)
    print("  📷 机械臂视觉摄像头快速检测工具")
    print("=" * 60)

    # 尝试检测可用的摄像头索引（0, 1, 2）
    available_cams = []
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                available_cams.append((idx, w, h))
            cap.release()

    if not available_cams:
        print("❌ 未检测到可用的摄像头，请检查 USB 接口是否插好！")
        return

    print("✅ 检测到以下可用摄像头设备:")
    for idx, w, h in available_cams:
        print(f"   -> [设备编号 {idx}]: 默认分辨率 {w} x {h}")

    selected_idx = available_cams[0][0]
    print(f"\n▶ 正在打开默认设备 [编号 {selected_idx}] 预览实时画面...")
    print("💡 操作提示:")
    print("   - 按键盘 【空格键 (Space)】: 截取当前机械臂桌面画面并保存")
    print("   - 按键盘 【ESC 键 或 Q 键】: 退出预览\n")

    cap = cv2.VideoCapture(selected_idx, cv2.CAP_DSHOW)
    # 尝试设置较高分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    save_dir = os.path.join(os.path.dirname(__file__), "captured_images")
    os.makedirs(save_dir, exist_ok=True)

    capture_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取画面帧失败。")
            break

        # 显示画面
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            "Press [SPACE] to Capture, [Q/ESC] to Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow("Robot Camera Preview", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') :
            capture_count += 1
            filename = f"table_view_{int(time.time())}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, frame)
            print(f"📸 抓取照片已保存至: {save_path}")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("预览已结束。")

if __name__ == "__main__":
    test_cameras()
