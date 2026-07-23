"""手勢與姿勢偵測：整合 MediaPipe Tasks 的 HandLandmarker + PoseLandmarker，
並用幾何規則（手指關節角度、手掌形狀、手與身體關鍵點的相對位置）判斷手勢標籤。

原本 6 種手勢對應的 meme 與動作描述（詳見 scripts/action.md）：
- pointing_finger      及川徹：手臂伸直、且食指伸直指向遠方，其他手指彎曲
- claw_reach           綠谷出久：單手四指微彎、朝外伸展（不是完全握拳也不是完全伸直）
- double_fist_stacked  布瑠部由良由良：雙手都握拳，在胸前一上一下疊在一起
- fist_up_down         敗者食塵：雙手都比類似 rock 的手勢（食指、小指伸直，中指、無名指彎曲），
                       一手手肘彎曲讓手朝上、另一手讓手朝下/朝前
- cover_one_eye        阿嬤特拉斯：單手五指張開放在臉前，剛好遮住一隻眼睛、另一隻眼睛露出
- josuke_pose          東方仗助：一手五指張開貼臉側，另一手五指伸直併攏貼在下巴附近

另外新增 5 種「領域展開」相關手勢：
- yuta_domain          乙骨優太（真贋相愛）：一手握拳（手掌朝下），另一手四指伸直朝外、拇指內縮
- sukuna_domain        宿儺：雙手手指緊密交疊，舉在嘴巴前面，形狀是手腕分開、指尖收攏的
                       三角錐狀（不是單手動作；因為手指遮擋嚴重，主要靠「手腕距離 > 指尖
                       距離」+「靠近嘴巴」的位置判斷，不特別檢查手指細節）
- hakari_domain        秤金次（坐殺博徒）：一手拇指+食指捏成圈（類似 OK 手勢，其餘三指伸直），
                       另一手在下方、四指伸直
- gojo_domain          五條悟（無量空處）：單手食指伸直，中指彎曲並與食指指尖捏合在一起，
                       無名指、小指彎曲收起（類似打彈指前的預備動作）
- reverse_cross_palms  反叉合掌（雙手合十、中指與無名指互相交叉的通用結印）：
                       雙手手腕緊貼在一起、且都不是握拳/rock 手勢
                       ⚠️ 這個手勢手指深度交疊，遮擋嚴重，用單一 RGB 鏡頭的關鍵點模型
                       很難精準辨識實際的手指交叉細節，這裡改用「雙手手腕貼在一起」做
                       近似判斷，穩定度會比其他手勢差，需要依實測狀況調整門檻。

注意：MediaPipe 從 0.10.31 版開始移除了舊版的 ``mp.solutions`` API，
所以這裡改用新版的 Tasks API（``mediapipe.tasks.python.vision``），
需要先執行 ``python scripts/download_models.py`` 下載模型檔（.task）到 models/ 資料夾。

這個版本不再使用 FaceLandmarker：「遮住一隻眼睛」「手貼臉側/下巴」等需要臉部參考點的
判斷，改用 PoseLandmarker 本身就有輸出的眼睛/耳朵/嘴角關鍵點（索引 1~10）做近似判斷，
精準度不如專門的臉部網格模型，但換來不需要額外載入一個模型檔。
"""
from __future__ import annotations

import itertools
import os

import mediapipe as mp

from utils.landmark_utils import angle_between, distance, midpoint

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# 每個手指對應的 (掌指關節, 近端關節, 遠端關節, 指尖) landmark 索引
FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FOUR_FINGERS = ("index", "middle", "ring", "pinky")

# 手指關節夾角門檻（單位：度）
STRAIGHT_ANGLE_THRESHOLD = 160.0   # >= 這個角度視為「伸直」
FIST_ANGLE_MAX = 100.0             # < 這個角度視為「完全彎曲」（緊握拳頭）
# claw（微彎，例如綠谷出久）的角度區間是 [FIST_ANGLE_MAX, STRAIGHT_ANGLE_THRESHOLD)，
# 也就是「不是緊握拳頭，但也還沒完全伸直」

# 手臂（肩-肘-腕）夾角 >= 這個角度，才視為「手臂伸直向前伸」（及川徹需要手臂伸直，用來跟綠谷出久的手掌區分）
ARM_STRAIGHT_ANGLE_MIN = 150.0

# 指尖間距（相對手掌大小正規化後）小於這個比例，視為「併攏」（flat_hand，例如貼下巴的手刀）
FLAT_HAND_SPREAD_MAX = 0.16

# 雙拳疊在一起（double_fist_stacked）：水平距離要夠近、垂直距離要有一定落差
STACK_MAX_X_GAP = 0.18
STACK_MIN_Y_GAP = 0.06

# 手肘到手腕的垂直距離（正規化座標）超過這個比例，才視為手臂明顯朝上或朝下
ARM_UP_DOWN_MIN_GAP = 0.05

# 判斷手掌是否蓋住眼睛時，手掌 bounding box 額外放寬的邊界
EYE_COVER_MARGIN = 0.02

# 判斷手是否貼近臉頰/耳朵、下巴/嘴角的距離門檻（正規化座標），故意放寬一點，
# 避免因鏡頭角度/手掌大小估算誤差導致 josuke_pose 判斷不出來
FACE_SIDE_MAX_DIST = 0.28
CHIN_MAX_DIST = 0.25

# 指尖捏合（拇指碰另一指指尖，例如 OK 手勢、彈指手勢）的距離門檻（相對手掌大小）
# 故意放寬一點（原本 0.35 太嚴格，實測手指沒有完全碰到指尖也常判斷不出來）
PINCH_MAX_RATIO = 0.42

# 宿儺手勢：食指、小指彎曲角度上限（> 這個角度就不算彎曲）；
# 中指、無名指的「伸直」門檻用比一般更寬鬆的數值（不用完全伸直到 160 度）；
# 以及中指、無名指指尖互碰的距離門檻（不用完全貼緊，靠近就算）
# 這三個門檻原本設太嚴格，實測很難剛好卡進區間，故意放寬一點
SUKUNA_BENT_ANGLE_MAX = 155.0
SUKUNA_STRAIGHT_MIN = 145.0
SUKUNA_TOUCH_MAX_RATIO = 0.25

# 反叉合掌：雙手手腕距離小於這個比例（正規化座標），視為「雙手合十貼在一起」
PALMS_TOUCH_MAX_DIST = 0.08

# 宿儺手勢（雙手版）：雙手手指緊密交疊舉在嘴巴前面，手腕距離門檻比反叉合掌寬鬆一點
# （兩手交疊時手腕本身不一定完全貼在一起），但額外要求「舉在嘴巴附近」
SUKUNA_HANDS_TOGETHER_MAX_DIST = 0.22

# 宿儺（手腕分開、指尖收攏的三角錐狀）跟反叉合掌（手腕、指尖都貼在一起的平行合掌狀）
# 光看「手腕距離」很容易搞混，因為兩者的手腕距離範圍會重疊。
# 關鍵差異是：宿儺的「手腕距離」會明顯大於「指尖距離」（上窄下寬），
# 反叉合掌則兩者差不多（整段都窄）。用這個差值當作宿儺的額外判斷條件。
SUKUNA_CONVERGENCE_MARGIN = 0.04

# Pose landmark 索引（MediaPipe Pose 33 點）
POSE_NOSE = 0
POSE_LEFT_EYE = 2
POSE_RIGHT_EYE = 5
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8
POSE_MOUTH_LEFT = 9
POSE_MOUTH_RIGHT = 10
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

# 除錯畫面用的關鍵點連線（Tasks API 不再內建 drawing_utils，改成手動列出常用連線）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


def _landmarks_to_xy(landmark_list):
    return [(lm.x, lm.y) for lm in landmark_list]


def _finger_angles(points) -> dict[str, float]:
    return {
        finger: angle_between(points[a], points[b], points[c])
        for finger, (a, b, c, _tip) in FINGER_JOINTS.items()
    }


def _fingertip_spread(points, hand_size: float) -> float:
    """回傳食指~小指相鄰指尖的平均間距（相對手掌大小），數值越大代表手指越張開。"""
    tip_pairs = ((8, 12), (12, 16), (16, 20))
    gaps = [distance(points[a], points[b]) / hand_size for a, b in tip_pairs]
    return sum(gaps) / len(gaps)


def _tip_distance_ratio(points, idx_a: int, idx_b: int, hand_size: float) -> float:
    """回傳兩個指尖 landmark 索引之間的距離（相對手掌大小），用來判斷是否「捏合/互碰」。"""
    return distance(points[idx_a], points[idx_b]) / hand_size


def _fingertip_centroid(points):
    """回傳五指指尖的重心點，近似這隻手手指延伸到的位置，
    比單獨看某一根指尖更能抵抗遮擋造成的單點誤判。"""
    tip_indices = (4, 8, 12, 16, 20)
    xs = sum(points[i][0] for i in tip_indices) / len(tip_indices)
    ys = sum(points[i][1] for i in tip_indices) / len(tip_indices)
    return (xs, ys)


def _classify_hand_shape(points) -> str | None:
    """依單手 21 個關鍵點，判斷手掌形狀（純形狀，不含位置資訊）。

    回傳值：
    - "ok_sign": 拇指、食指指尖捏合成一個圈，中指、無名指、小指都伸直（秤金次「坐殺博徒」用）
    - "snap_pinch": 食指伸直，中指彎曲並與食指指尖捏合在一起，無名指、小指彎曲收起
      （五條悟「無量空處」用；跟 ok_sign 的差別在捏合的是食指+中指而不是拇指+食指）
    - "sukuna_mudra": 食指、小指彎曲，中指、無名指伸直且指尖互碰（宿儺用）
    - "rock_horns": 食指、小指伸直，中指、無名指彎曲（類似 rock 手勢，敗者食塵用）
    - "pointing": 只有食指伸直，其他三指都彎曲（及川徹用，另外還需搭配手臂伸直，見 _is_arm_extended）
    - "fist": 四指（食指~小指）都「完全彎曲」（角度 < FIST_ANGLE_MAX），視為緊握拳頭
    - "open_palm": 四指都伸直、且指尖之間有張開間距
    - "flat_hand": 四指都伸直、但指尖併攏（沒有張開）
    - "claw": 四指都在「微彎」區間 [FIST_ANGLE_MAX, STRAIGHT_ANGLE_THRESHOLD)
      （不是完全伸直也不是完全握拳，例如綠谷出久的手勢）
    - None: 不符合以上任何形狀
    """
    angles = _finger_angles(points)
    straight = {finger: angles[finger] >= STRAIGHT_ANGLE_THRESHOLD for finger in FINGER_JOINTS}
    fist_tight = {finger: angles[finger] < FIST_ANGLE_MAX for finger in FOUR_FINGERS}
    hand_size = distance(points[0], points[9]) or 1e-6

    thumb_index_ratio = _tip_distance_ratio(points, 4, 8, hand_size)
    index_middle_ratio = _tip_distance_ratio(points, 8, 12, hand_size)
    middle_ring_ratio = _tip_distance_ratio(points, 12, 16, hand_size)

    if thumb_index_ratio < PINCH_MAX_RATIO and straight["middle"] and straight["ring"] and straight["pinky"]:
        return "ok_sign"

    if (
        straight["index"]
        and index_middle_ratio < PINCH_MAX_RATIO
        and not straight["middle"]
        and not straight["ring"]
        and not straight["pinky"]
    ):
        return "snap_pinch"

    if (
        angles["index"] < SUKUNA_BENT_ANGLE_MAX
        and angles["pinky"] < SUKUNA_BENT_ANGLE_MAX
        and angles["middle"] >= SUKUNA_STRAIGHT_MIN
        and angles["ring"] >= SUKUNA_STRAIGHT_MIN
        and middle_ring_ratio < SUKUNA_TOUCH_MAX_RATIO
    ):
        return "sukuna_mudra"

    if straight["index"] and straight["pinky"] and not straight["middle"] and not straight["ring"]:
        return "rock_horns"

    if straight["index"] and not any(straight[f] for f in ("middle", "ring", "pinky")):
        return "pointing"

    if all(fist_tight[f] for f in FOUR_FINGERS):
        return "fist"

    if all(straight[f] for f in FOUR_FINGERS):
        spread = _fingertip_spread(points, hand_size)
        return "flat_hand" if spread < FLAT_HAND_SPREAD_MAX else "open_palm"

    if all(not straight[f] and not fist_tight[f] for f in FOUR_FINGERS):
        return "claw"

    return None


def _match_side(wrist, pose_points) -> str:
    """依手腕座標，回傳這隻手比較靠近身體的哪一側（"left" 或 "right"）。"""
    left_wrist, right_wrist = pose_points[POSE_LEFT_WRIST], pose_points[POSE_RIGHT_WRIST]
    return "left" if distance(wrist, left_wrist) <= distance(wrist, right_wrist) else "right"


def _match_elbow(wrist, pose_points):
    """依手腕座標，回傳距離較近的那一側 Pose 手肘座標（用來判斷該手所屬的手臂方向）。"""
    side = _match_side(wrist, pose_points)
    return pose_points[POSE_LEFT_ELBOW] if side == "left" else pose_points[POSE_RIGHT_ELBOW]


def _is_arm_extended(wrist, pose_points) -> bool:
    """判斷這隻手所屬的手臂是否伸直向前伸（肩-肘-腕夾角接近 180 度）。"""
    side = _match_side(wrist, pose_points)
    shoulder = pose_points[POSE_LEFT_SHOULDER if side == "left" else POSE_RIGHT_SHOULDER]
    elbow = pose_points[POSE_LEFT_ELBOW if side == "left" else POSE_RIGHT_ELBOW]
    return angle_between(shoulder, elbow, wrist) >= ARM_STRAIGHT_ANGLE_MIN


def _arm_direction(wrist, elbow) -> str | None:
    """回傳手臂方向："up"（手腕明顯高於手肘）、"down"（手腕明顯低於手肘）或 None（不明顯）。"""
    dy = wrist[1] - elbow[1]  # 螢幕座標原點在左上角，y 越小代表越靠畫面上方
    if dy < -ARM_UP_DOWN_MIN_GAP:
        return "up"
    if dy > ARM_UP_DOWN_MIN_GAP:
        return "down"
    return None


def _is_stacked(wrist_a, wrist_b) -> bool:
    """判斷兩手腕是否「水平靠近、垂直有落差」，即一上一下疊在一起（不是左右並排）。"""
    x_gap = abs(wrist_a[0] - wrist_b[0])
    y_gap = abs(wrist_a[1] - wrist_b[1])
    return x_gap < STACK_MAX_X_GAP and y_gap >= STACK_MIN_Y_GAP


def _is_one_up_one_down(wrist_a, wrist_b, pose_points) -> bool:
    dir_a = _arm_direction(wrist_a, _match_elbow(wrist_a, pose_points))
    dir_b = _arm_direction(wrist_b, _match_elbow(wrist_b, pose_points))
    return {dir_a, dir_b} == {"up", "down"}


def _is_palms_touching(wrist_a, wrist_b) -> bool:
    """判斷兩手腕是否緊貼在一起（反叉合掌的雙手手掌根部會貼在一起）。"""
    return distance(wrist_a, wrist_b) < PALMS_TOUCH_MAX_DIST


def _covers_one_eye(hand_points, pose_points) -> bool:
    """判斷這隻手的 bounding box 是否剛好蓋住其中一隻眼睛、另一隻眼睛沒被蓋住。"""
    xs = [p[0] for p in hand_points]
    ys = [p[1] for p in hand_points]
    min_x, max_x = min(xs) - EYE_COVER_MARGIN, max(xs) + EYE_COVER_MARGIN
    min_y, max_y = min(ys) - EYE_COVER_MARGIN, max(ys) + EYE_COVER_MARGIN

    def _covered(eye) -> bool:
        return min_x <= eye[0] <= max_x and min_y <= eye[1] <= max_y

    return _covered(pose_points[POSE_LEFT_EYE]) != _covered(pose_points[POSE_RIGHT_EYE])


def _near_face_side(wrist, pose_points) -> bool:
    left_ear, right_ear = pose_points[POSE_LEFT_EAR], pose_points[POSE_RIGHT_EAR]
    return min(distance(wrist, left_ear), distance(wrist, right_ear)) < FACE_SIDE_MAX_DIST


def _near_chin(wrist, pose_points) -> bool:
    mouth_center = midpoint(pose_points[POSE_MOUTH_LEFT], pose_points[POSE_MOUTH_RIGHT])
    return distance(wrist, mouth_center) < CHIN_MAX_DIST


class GestureDetector:
    """整合手部與姿勢偵測，回傳目前畫面判斷到的手勢標籤（可能為 None）。

    偵測優先序：
    1. 雙手同時符合的組合手勢（依序檢查
       fist_up_down > double_fist_stacked > yuta_domain > hakari_domain >
       sukuna_domain > reverse_cross_palms > josuke_pose）
    2. 單手形狀手勢（pointing_finger > claw_reach > sukuna_domain(備援) > gojo_domain > cover_one_eye）

    pointing_finger 與 claw_reach 容易搞混（因為 claw 手勢的手指可能微伸到接近伸直），
    所以 pointing_finger 額外要求「手臂伸直」（肩-肘-腕夾角 >= ARM_STRAIGHT_ANGLE_MIN），
    claw_reach 則不限制手臂角度，只看手指彎曲程度。

    cover_one_eye 與 josuke_pose 也容易搞混（兩者都會有一隻張開的手靠近臉），
    所以 cover_one_eye 限制「只有偵測到剛好一隻手」時才會判斷，雙手都在畫面裡時
    一律先看是否符合 josuke_pose。

    ok_sign（秤金次用）、snap_pinch（五條悟用）三種捏合形狀，靠捏合的是哪根手指、
    其餘手指是否伸直來互相區分，理論上不會混淆，但實際測試若鏡頭角度不佳導致誤判，
    可調整 PINCH_MAX_RATIO 等門檻。

    sukuna_domain（宿儺）實際是「雙手」一起交疊舉在嘴巴前面的手勢，不是單手動作；
    因為雙手手指緊密交疊時，21 點手部關鍵點模型很容易因為遮擋誤判每隻手的手指角度，
    所以主要判斷方式改成幾何近似：手腕距離、指尖距離、跟嘴巴的距離。
    sukuna_domain 跟 reverse_cross_palms 都是「雙手貼在一起舉高」的姿勢，光看手腕距離
    很容易搞混（兩者的手腕距離範圍會重疊），所以宿儺額外要求「手腕距離明顯大於指尖
    距離」（手腕分開、指尖收攏的三角錐狀）+「靠近嘴巴」，反叉合掌則是手腕跟指尖都
    貼在一起的平行合掌狀，兩者形狀本質不同，用這個差異來區分，不是單純比手腕距離。
    sukuna_mudra 這個單手形狀/判斷保留當備援（萬一鏡頭只清楚拍到其中一隻手），
    但不是主要判斷路徑。
    """

    def __init__(
        self,
        hand_model_path: str,
        pose_model_path: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ):
        for path, name in ((hand_model_path, "手勢(Hand)"), (pose_model_path, "姿勢(Pose)")):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"找不到{name}偵測模型檔: {path}\n"
                    "請先在專案根目錄執行「python scripts/download_models.py」下載模型。"
                )

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        # 給 main.py 除錯畫關鍵點/顯示文字用
        self.last_hand_landmarks = []
        self.last_pose_landmarks = None
        self.last_hand_shapes: list[str | None] = []

    def detect(self, mp_image: mp.Image, timestamp_ms: int) -> str | None:
        """輸入 mediapipe Image 與遞增的時間戳（毫秒），回傳偵測到的手勢標籤。"""
        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        self.last_hand_landmarks = hand_result.hand_landmarks or []

        # 沒有偵測到手時，跳過 Pose 偵測（省一個模型的運算量）：
        # 反正所有手勢都至少需要一隻手，沒手就不可能觸發任何手勢，Pose 資訊也用不到
        if self.last_hand_landmarks:
            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            self.last_pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        else:
            self.last_pose_landmarks = None
        pose_points = _landmarks_to_xy(self.last_pose_landmarks) if self.last_pose_landmarks else None

        hands = []
        for landmarks in self.last_hand_landmarks:
            points = _landmarks_to_xy(landmarks)
            hands.append({"shape": _classify_hand_shape(points), "points": points, "wrist": points[0]})
        self.last_hand_shapes = [hand["shape"] for hand in hands]

        label = None
        if len(hands) >= 2:
            label = self._classify_two_hand_gesture(hands, pose_points)
        if label is None:
            label = self._classify_single_hand_gestures(hands, pose_points)
        return label

    @staticmethod
    def _classify_two_hand_gesture(hands, pose_points) -> str | None:
        for hand_a, hand_b in itertools.combinations(hands, 2):
            both_rock = hand_a["shape"] == "rock_horns" and hand_b["shape"] == "rock_horns"
            if both_rock and pose_points and _is_one_up_one_down(hand_a["wrist"], hand_b["wrist"], pose_points):
                return "fist_up_down"

            both_fists = hand_a["shape"] == "fist" and hand_b["shape"] == "fist"
            if both_fists and _is_stacked(hand_a["wrist"], hand_b["wrist"]):
                return "double_fist_stacked"

            # 乙骨優太「真贋相愛」：一手握拳、另一手四指伸直（不特別限制相對位置/是否併攏）
            for fist_hand, open_hand in ((hand_a, hand_b), (hand_b, hand_a)):
                if fist_hand["shape"] == "fist" and open_hand["shape"] in ("open_palm", "flat_hand"):
                    return "yuta_domain"

            # 秤金次「坐殺博徒」：一手比 OK 手勢（在上）、另一手四指伸直在下方
            for ok_hand, straight_hand in ((hand_a, hand_b), (hand_b, hand_a)):
                if (
                    ok_hand["shape"] == "ok_sign"
                    and straight_hand["shape"] in ("open_palm", "flat_hand")
                    and straight_hand["wrist"][1] > ok_hand["wrist"][1]
                ):
                    return "hakari_domain"

            # 宿儺：雙手手指緊密交疊、舉在嘴巴前面，形狀是「手腕分開、指尖收攏」的三角錐狀
            # （手腕距離明顯大於指尖距離），用這點跟手腕/指尖都貼在一起的 reverse_cross_palms
            # 做區分，而不是單純比較手腕距離（因為兩者的手腕距離範圍其實會重疊，
            # 容易讓反叉合掌被誤判成宿儺）。不特別要求手指的精確形狀，因為兩手手指緊密
            # 交疊時，關鍵點模型的角度判斷常常被遮擋誤導，很難準確算出來。
            wrist_gap = distance(hand_a["wrist"], hand_b["wrist"])
            fingertip_gap = distance(
                _fingertip_centroid(hand_a["points"]), _fingertip_centroid(hand_b["points"])
            )
            if (
                hand_a["shape"] not in ("fist", "rock_horns")
                and hand_b["shape"] not in ("fist", "rock_horns")
                and wrist_gap < SUKUNA_HANDS_TOGETHER_MAX_DIST
                and wrist_gap - fingertip_gap > SUKUNA_CONVERGENCE_MARGIN
                and pose_points
                and _near_chin(midpoint(hand_a["wrist"], hand_b["wrist"]), pose_points)
            ):
                return "sukuna_domain"

            # 反叉合掌：雙手手腕緊貼在一起（合十），且都不是握拳/rock 手勢
            # （fist_up_down、double_fist_stacked 已經在上面優先判斷過，這裡排除避免搶先觸發）
            if (
                hand_a["shape"] not in ("fist", "rock_horns")
                and hand_b["shape"] not in ("fist", "rock_horns")
                and _is_palms_touching(hand_a["wrist"], hand_b["wrist"])
            ):
                return "reverse_cross_palms"

            # 下巴那隻手不特別區分「併攏(flat_hand)」還是「張開(open_palm)」，
            # 只要五指是伸直的即可（實際測試發現嚴格要求併攏反而常常判斷不出來）
            for ear_hand, chin_hand in ((hand_a, hand_b), (hand_b, hand_a)):
                if (
                    ear_hand["shape"] == "open_palm"
                    and chin_hand["shape"] in ("open_palm", "flat_hand")
                    and pose_points
                    and _near_face_side(ear_hand["wrist"], pose_points)
                    and _near_chin(chin_hand["wrist"], pose_points)
                ):
                    return "josuke_pose"
        return None

    @staticmethod
    def _classify_single_hand_gestures(hands, pose_points) -> str | None:
        for hand in hands:
            if hand["shape"] == "pointing" and pose_points and _is_arm_extended(hand["wrist"], pose_points):
                return "pointing_finger"
            if hand["shape"] == "claw":
                return "claw_reach"
            if hand["shape"] == "sukuna_mudra":
                return "sukuna_domain"
            if hand["shape"] == "snap_pinch":
                return "gojo_domain"
            # cover_one_eye（阿嬤特拉斯）本身是「單手」動作，這裡限制只有偵測到剛好一隻手時才判斷，
            # 避免在雙手比 josuke_pose（東方仗助）但角度沒對齊時，被誤判成 cover_one_eye
            if (
                len(hands) == 1
                and hand["shape"] == "open_palm"
                and pose_points
                and _covers_one_eye(hand["points"], pose_points)
            ):
                return "cover_one_eye"
        return None

    def close(self):
        self._hand_landmarker.close()
        self._pose_landmarker.close()
