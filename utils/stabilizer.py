"""去抖動 + 冷卻邏輯：避免偵測結果閃爍或同一個 meme 連續彈出。"""
from __future__ import annotations

import time


class Stabilizer:
    """將「每一帳的原始偵測標籤」轉換成「真正應該觸發的標籤」。

    規則：
    1. 同一個標籤要連續出現 ``hold_frames`` 帳以上才算「確認」，避免單帳誤判造成閃爍。
    2. 同一個標籤觸發後，需要等待 ``cooldown_seconds`` 秒才能再次觸發同一個標籤。
    3. 任何一次觸發後，至少要等待 ``switch_cooldown_seconds`` 秒才能觸發下一個（即使是不同標籤），
       避免手勢/表情切換瞬間造成連續彈出。
    """

    def __init__(
        self,
        hold_frames: int = 6,
        cooldown_seconds: float = 2.5,
        switch_cooldown_seconds: float = 0.5,
    ):
        self.hold_frames = hold_frames
        self.cooldown_seconds = cooldown_seconds
        self.switch_cooldown_seconds = switch_cooldown_seconds

        self._candidate: str | None = None
        self._candidate_count = 0
        self._last_trigger_time: dict[str, float] = {}
        self._last_switch_time = 0.0

    def update(self, label: str | None) -> str | None:
        """輸入本帳偵測到的標籤（可能是 None），回傳本帳「應該觸發」的標籤或 None。"""
        now = time.time()

        if label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = label
            self._candidate_count = 1

        if label is None or self._candidate_count < self.hold_frames:
            return None

        if now - self._last_trigger_time.get(label, 0.0) < self.cooldown_seconds:
            return None

        if now - self._last_switch_time < self.switch_cooldown_seconds:
            return None

        self._last_trigger_time[label] = now
        self._last_switch_time = now
        return label
