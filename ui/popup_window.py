"""負責彈出/自動關閉 meme 圖片小視窗的 Tkinter 元件。"""
from __future__ import annotations

import tkinter as tk

from PIL import Image, ImageTk

POSITION_MARGIN = 20


class PopupWindow:
    """一個置頂、無邊框的小視窗，用來顯示觸發後的 meme 圖片，並在一段時間後自動隱藏。"""

    def __init__(
        self,
        root: tk.Tk,
        width: int = 1000,
        height: int = 1000,
        duration_seconds: float = 3.0,
        position: str = "center",
    ):
        self.root = root
        self.width = width
        self.height = height
        self.duration_seconds = duration_seconds
        self.position = position

        self.toplevel = tk.Toplevel(root)
        self.toplevel.overrideredirect(True)  # 不顯示標題列/邊框
        self.toplevel.attributes("-topmost", True)
        self.toplevel.configure(bg="black")
        self.toplevel.withdraw()  # 一開始先隱藏

        self.label = tk.Label(self.toplevel, bd=0)
        self.label.pack()

        self._photo = None  # 保留參考，避免被垃圾回收
        self._hide_job = None
        self._place_window()

    def _place_window(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        positions = {
            "top-right": (screen_w - self.width - POSITION_MARGIN, POSITION_MARGIN),
            "top-left": (POSITION_MARGIN, POSITION_MARGIN),
            "bottom-right": (screen_w - self.width - POSITION_MARGIN, screen_h - self.height - POSITION_MARGIN),
            "bottom-left": (POSITION_MARGIN, screen_h - self.height - POSITION_MARGIN),
            "center": ((screen_w - self.width) // 2, (screen_h - self.height) // 2),
            # 螢幕右半邊、垂直方向置中：水平位置是「右半邊的正中間」（畫面 3/4 寬度處），不是貼右邊緣
            "center-right": (int(screen_w * 0.75 - self.width / 2), (screen_h - self.height) // 2),
        }
        x, y = positions.get(self.position, positions["top-right"])
        self.toplevel.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def show(self, image_path: str):
        try:
            image = Image.open(image_path)
        except (FileNotFoundError, OSError) as exc:
            print(f"[popup_window] 找不到或無法開啟圖片: {image_path} ({exc})")
            return

        image = image.convert("RGBA")
        image.thumbnail((self.width, self.height))
        self._photo = ImageTk.PhotoImage(image)
        self.label.configure(image=self._photo)

        self._place_window()
        self.toplevel.deiconify()
        self.toplevel.lift()

        if self._hide_job is not None:
            self.root.after_cancel(self._hide_job)
        self._hide_job = self.root.after(int(self.duration_seconds * 1000), self.hide)

    def hide(self):
        self.toplevel.withdraw()
        self._hide_job = None
