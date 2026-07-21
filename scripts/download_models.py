"""下載 MediaPipe Tasks 需要的模型檔（.task）到 models/ 資料夾。

執行方式（在專案根目錄下）：
    python scripts/download_models.py

只需要執行一次、需要網路連線；下載完成後 main.py 即可完全離線執行。
若已經下載過（檔案已存在），會自動跳過。
"""
from __future__ import annotations

import os
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def download(filename: str, url: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest_path):
        print(f"已存在，跳過: {dest_path}")
        return
    print(f"下載中: {url}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"完成: {dest_path}")


def main():
    had_error = False
    for filename, url in MODEL_URLS.items():
        try:
            download(filename, url)
        except Exception as exc:  # noqa: BLE001 - 讓使用者看到明確錯誤原因，而不是整個程式中斷
            had_error = True
            print(f"下載失敗: {filename} ({exc})")

    if had_error:
        print(
            "\n若下載失敗，可能是網路連線問題，"
            "也可以改用瀏覽器直接開啟下方網址手動下載，"
            "存到專案的 models/ 資料夾並確認檔名一致：\n"
            + "\n".join(MODEL_URLS.values())
        )


if __name__ == "__main__":
    main()
