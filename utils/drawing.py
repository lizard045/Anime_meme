"""簡易的關鍵點除錯繪製工具（純用 OpenCV 畫圓/畫線）。

MediaPipe 從 0.10.31 版開始移除了舊版 ``mp.solutions.drawing_utils``，
新版 Tasks API 只回傳關鍵點座標，畫面繪製需要自己實作，所以獨立寫在這裡。
"""
from __future__ import annotations

import cv2


def draw_points(frame, landmarks, color=(0, 255, 0), radius=2):
    """在 frame 上畫出每個 landmark 的位置（landmark.x/y 為 0~1 的正規化座標）。"""
    height, width = frame.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_connections(frame, landmarks, connections, color=(0, 255, 0), thickness=2):
    """依照 connections（(起點索引, 終點索引) 列表）在 landmarks 之間畫線。"""
    height, width = frame.shape[:2]
    for start_idx, end_idx in connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, color, thickness)
