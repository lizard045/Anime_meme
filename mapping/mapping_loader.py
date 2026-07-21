"""載入並查詢「偵測標籤 -> meme 圖片路徑」的對照表（mapping.json）。"""
from __future__ import annotations

import json
import os


class MappingLoader:
    """讀取 mapping.json，並提供標籤到圖片絕對路徑的查詢。

    每個標籤固定對應一張圖片（1:1），找不到對應標籤時會退回 "default" 項目；
    若連 default 都沒有設定則回傳 None。
    """

    def __init__(self, mapping_path: str, base_dir: str | None = None):
        self.mapping_path = mapping_path
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(mapping_path)))
        self._mapping: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self._mapping = json.load(f)

    def get_image_path(self, label: str) -> str | None:
        relative_path = self._mapping.get(label) or self._mapping.get("default")
        if relative_path is None:
            return None
        return os.path.join(self.base_dir, relative_path)

    @property
    def labels(self):
        return list(self._mapping.keys())
