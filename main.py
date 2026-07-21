"""攝影機動作辨識 Meme 顯示工具 - 主程式進入點。

執行方式（第一次使用前，記得先執行 python scripts/download_models.py 下載模型檔）：
    python main.py

會開啟一個顯示攝影機畫面的小視窗（含即時偵測結果文字），
偵測到特定手勢時，會在螢幕角落彈出對應的 meme 圖片。
關閉視窗或按下 ESC 即可結束程式。
"""
from __future__ import annotations

import os
import time
import tkinter as tk

import cv2
import mediapipe as mp
import yaml
from PIL import Image, ImageTk

from detectors.gesture_detector import HAND_CONNECTIONS, POSE_CONNECTIONS, GestureDetector
from mapping.mapping_loader import MappingLoader
from ui.popup_window import PopupWindow
from utils.drawing import draw_connections, draw_points
from utils.stabilizer import Stabilizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class _TimestampProvider:
    """提供給 MediaPipe Tasks VIDEO 模式使用的、保證嚴格遞增的毫秒時間戳。"""

    def __init__(self):
        self._start = time.time()
        self._last = -1

    def next(self) -> int:
        ts = int((time.time() - self._start) * 1000)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        return ts


class App:
    def __init__(self, config: dict):
        self.config = config

        camera_cfg = config.get("camera", {})
        self.mirror = camera_cfg.get("mirror", True)
        self.frame_width = camera_cfg.get("width", 960)
        self.frame_height = camera_cfg.get("height", 540)

        # Windows 上預設的 MSMF 後端常常會忽略指定的解析度/FPS、自動幫你降級，
        # 改用 CAP_DSHOW 後端對解析度/FPS 設定的支援通常比較準確
        self.cap = cv2.VideoCapture(camera_cfg.get("index", 0), cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, camera_cfg.get("fps", 30))
        # 大多數 webcam 預設會用 MJPG/YUY2 等格式，FourCC 設成 MJPG 通常能讓鏡頭跑到較高 FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"無法開啟攝影機（index={camera_cfg.get('index', 0)}），"
                "請確認鏡頭是否被其他程式占用，或嘗試修改 config.yaml 的 camera.index。"
            )

        detection_cfg = config.get("detection", {})
        # 手勢/姿勢偵測用的畫面寬度：縮小輸入解析度可以大幅降低 MediaPipe 的運算量，
        # 且不影響偵測準確度太多（landmark 座標本身是 0~1 正規化值，跟輸入解析度無關）；
        # 畫面顯示與除錯關鍵點繪製仍然使用原始解析度，不受影響。
        self.detection_max_width = detection_cfg.get("max_width", 480)
        models_cfg = config.get("models", {})
        self.gesture_enabled = config.get("gesture", {}).get("enabled", True)

        self.gesture_detector = (
            GestureDetector(
                hand_model_path=os.path.join(BASE_DIR, models_cfg.get("hand_landmarker_path", "models/hand_landmarker.task")),
                pose_model_path=os.path.join(BASE_DIR, models_cfg.get("pose_landmarker_path", "models/pose_landmarker_lite.task")),
                min_detection_confidence=detection_cfg.get("min_detection_confidence", 0.5),
                min_tracking_confidence=detection_cfg.get("min_tracking_confidence", 0.5),
            )
            if self.gesture_enabled
            else None
        )

        stabilizer_cfg = config.get("stabilizer", {})
        self.stabilizer = Stabilizer(
            hold_frames=stabilizer_cfg.get("hold_frames", 6),
            cooldown_seconds=stabilizer_cfg.get("cooldown_seconds", 2.5),
            switch_cooldown_seconds=stabilizer_cfg.get("switch_cooldown_seconds", 0.5),
        )

        mapping_path = os.path.join(BASE_DIR, "mapping", "mapping.json")
        self.mapping_loader = MappingLoader(mapping_path, base_dir=BASE_DIR)

        debug_cfg = config.get("debug", {})
        self.show_metrics = debug_cfg.get("show_metrics", True)
        self.draw_landmarks = debug_cfg.get("draw_landmarks", True)

        self._timestamps = _TimestampProvider()

        self.root = tk.Tk()
        self.root.title("攝影機手勢 Meme 小工具")
        self.root.bind("<Escape>", lambda _event: self._on_close())

        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        self.status_label = tk.Label(
            self.root, text="準備中...", font=("Microsoft JhengHei", 11), anchor="w", justify="left"
        )
        self.status_label.pack(fill="x")

        popup_cfg = config.get("popup", {})
        self.popup = PopupWindow(
            self.root,
            width=popup_cfg.get("width", 300),
            height=popup_cfg.get("height", 300),
            duration_seconds=popup_cfg.get("duration_seconds", 3.0),
            position=popup_cfg.get("position", "top-right"),
        )

        self._photo_image = None
        self._last_frame_time = time.time()
        self._fps = 0.0
        self._closed = False

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        if self.gesture_detector:
            self.gesture_detector.close()
        self.cap.release()
        self.root.destroy()

    def _draw_debug_landmarks(self, rgb_frame):
        """在（RGB 通道排列的）畫面上畫關鍵點，顏色用 RGB 順序指定。"""
        if not self.gesture_detector:
            return
        for hand_landmarks in self.gesture_detector.last_hand_landmarks:
            draw_connections(rgb_frame, hand_landmarks, HAND_CONNECTIONS, color=(255, 200, 0))
            draw_points(rgb_frame, hand_landmarks, color=(0, 255, 0))
        if self.gesture_detector.last_pose_landmarks:
            draw_connections(rgb_frame, self.gesture_detector.last_pose_landmarks, POSE_CONNECTIONS, color=(0, 200, 255))

    def _build_status_text(self, gesture_label) -> str:
        text = f"FPS: {self._fps:0.1f}  手勢: {gesture_label or '-'}"
        if self.show_metrics and self.gesture_detector:
            shapes = self.gesture_detector.last_hand_shapes
            text += f"\n單手形狀: {shapes if shapes else '-'}"
        return text

    def _update_frame(self):
        if self._closed:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return

        if self.mirror:
            frame = cv2.flip(frame, 1)

        # 只做一次 BGR->RGB 轉換：後面除錯關鍵點直接畫在這張 RGB 畫面上，
        # 顯示到 Tkinter 視窗時也直接沿用，不用再轉第二次
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 給模型偵測用的縮小版畫面（landmark 座標是 0~1 正規化值，縮小輸入不影響座標意義，
        # 只是讓 MediaPipe 運算量變小、跑得更快）
        height, width = rgb_frame.shape[:2]
        if width > self.detection_max_width:
            scale = self.detection_max_width / width
            detect_frame = cv2.resize(rgb_frame, (self.detection_max_width, int(height * scale)))
        else:
            detect_frame = rgb_frame

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_frame)
        timestamp_ms = self._timestamps.next()

        gesture_label = self.gesture_detector.detect(mp_image, timestamp_ms) if self.gesture_detector else None

        if self.draw_landmarks:
            self._draw_debug_landmarks(rgb_frame)

        triggered_label = self.stabilizer.update(gesture_label)

        if triggered_label:
            image_path = self.mapping_loader.get_image_path(triggered_label)
            if image_path:
                self.popup.show(image_path)
            print(f"[main] 觸發 meme: {triggered_label}")

        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        self.status_label.configure(text=self._build_status_text(gesture_label))

        image = Image.fromarray(rgb_frame)
        self._photo_image = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self._photo_image)

        # 只留 1ms 的最小間隔，讓實際更新速度由「鏡頭讀取 + 模型推論」的真實耗時決定，
        # 不要被這裡的排程延遲額外拖慢
        self.root.after(1, self._update_frame)

    def run(self):
        self._update_frame()
        self.root.mainloop()


def main():
    config = load_config()
    app = App(config)
    try:
        app.run()
    finally:
        app._on_close()


if __name__ == "__main__":
    main()
