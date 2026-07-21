"""共用的幾何計算工具，供 detectors 內的規則分類器使用。"""
from __future__ import annotations

import math

Point = tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """回傳兩個 (x, y) 座標之間的歐幾里得距離。"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1: Point, p2: Point) -> Point:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def angle_between(a: Point, b: Point, c: Point) -> float:
    """回傳以 b 為頂點、a-b-c 三點形成的夾角（單位：度）。

    用於判斷手指關節是伸直（角度接近 180 度）還是彎曲（角度明顯變小）。
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba * mag_bc == 0:
        return 180.0
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))
