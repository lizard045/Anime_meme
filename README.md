# AnimeMeme - 攝影機動作辨識 Meme 顯示工具

透過電腦攝影機即時偵測「手勢/身體姿勢」，偵測到特定動作時，
在螢幕角落彈出對應的謎因（meme）圖片。目前是一個在本機直接執行的 Python 小工具，
未來可視需求再擴充成網頁版。

> 目前版本已移除臉部表情偵測功能，改成 11 種依實際 meme 圖片手勢設計的專屬動作辨識
> （只用 `HandLandmarker` + `PoseLandmarker`，不再需要 `FaceLandmarker`）。

## 功能總覽：11 種手勢動作

每種手勢都是照對應 meme 圖片裡實際的手部/身體動作設計，不是隨便挑一個相近手勢代替：

| 標籤 | 對應 meme | 動作描述 |
| --- | --- | --- |
| `pointing_finger` | 及川徹 | 手臂伸直向前伸、且食指伸直指向遠方，其他手指彎曲 |
| `claw_reach` | 綠谷出久 | 單手四指（食指~小指）微彎、朝外伸展，不是完全握拳也不是完全伸直（不限制手臂角度） |
| `double_fist_stacked` | 布瑠部由良由良 | 雙手都握拳，在胸前一上一下疊在一起（不是左右並排） |
| `fist_up_down` | 敗者食塵 | 雙手都比類似 rock 的手勢（食指、小指伸直，中指、無名指彎曲），一手手肘彎曲讓手朝上舉高、另一手讓手朝下/朝前 |
| `cover_one_eye` | 阿嬤特拉斯（佐助血輪眼） | 單手五指張開放在臉前，剛好遮住一隻眼睛、另一隻眼睛露出 |
| `josuke_pose` | 東方仗助 | 雙手同時比出不同形狀：一手五指張開貼在臉的一側（太陽穴/耳朵附近），另一手五指伸直併攏貼在下巴附近 |
| `yuta_domain` | 乙骨優太（真贋相愛） | 一手握拳（手掌朝下），另一手四指伸直朝外、拇指內縮（不限制兩手相對位置） |
| `sukuna_domain` | 宿儺 | 雙手手指緊密交疊，舉在嘴巴前面（雙手手勢，不是單手；⚠️ 手指遮擋嚴重，主要用「雙手貼在一起 + 靠近嘴巴」的位置判斷） |
| `hakari_domain` | 秤金次（坐殺博徒） | 一手拇指+食指捏成一個圈（類似 OK 手勢，其餘三指伸直），另一手在下方、四指伸直 |
| `gojo_domain` | 五條悟（無量空處） | 單手食指伸直，中指彎曲並與食指指尖捏合在一起，無名指、小指彎曲收起 |
| `reverse_cross_palms` | 反叉合掌 | 雙手手腕緊貼在一起（合十），且都不是握拳/rock 手勢；⚠️ 這個手勢手指深度交疊，遮擋嚴重，實際只用「雙手貼在一起」做近似判斷，精準度較低 |

> `pointing_finger`（及川徹）與 `claw_reach`（綠谷出久）容易搞混，因為兩者都可能有手指
> 微微伸直的情況；所以特別把 `pointing_finger` 加上「手臂需伸直向前伸」的額外條件，
> 用來跟手臂不需伸直的 `claw_reach` 做區分。
>
> `ok_sign`（秤金次）、`snap_pinch`（五條悟）、`sukuna_mudra`（宿儺）三種形狀都是
> 「手指捏合」的變化：捏合的手指（食指/中指/無需捏合）跟其餘手指是否伸直不同，
> 理論上可以互相區分，但都是「靠近臉部的手部特寫」動作，鏡頭角度不佳時仍可能誤判，
> 需要依實測調整 `PINCH_MAX_RATIO`、`SUKUNA_BENT_ANGLE_MAX` 等門檻
> （見 `detectors/gesture_detector.py`）。

- 偵測到後會依照 `mapping/mapping.json` 對照表，在螢幕角落彈出對應圖片，幾秒後自動消失
- 內建去抖動/冷卻機制，避免畫面閃爍或同一張圖連續彈出
- 使用 MediaPipe 官方最新的 Tasks API（`HandLandmarker` / `PoseLandmarker`），
  手勢判斷全部用手指關節角度、手掌/手指相對位置等幾何規則計算
  （需要在第一次使用前下載 2 個模型檔，詳見下方安裝步驟，下載一次後即可離線使用）

## 環境需求

- Windows 10/11
- Python 3.9 ~ 3.11（建議 3.10 或 3.11，mediapipe 對新版 Python 的支援可能較慢跟上）
- 一台可用的攝影機（內建鏡頭或外接 USB 鏡頭皆可）

## 安裝步驟

在專案根目錄（`AnimeMeme/`）開啟 PowerShell：

```powershell
# 1. 建立並啟用虛擬環境（建議，避免污染系統 Python）
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. 安裝相依套件
pip install -r requirements.txt

# 3. 下載 MediaPipe 模型檔（只需要執行一次，需要網路連線）
python scripts/download_models.py

# 4. 執行主程式（images/ 內已經放好 11 張對應的 meme 圖片，不需要額外產生預留圖）
python main.py
```

> 模型下載說明：`download_models.py` 會把 `hand_landmarker.task`、`pose_landmarker_lite.task`
> 兩個檔案下載到 `models/` 資料夾（總共約 10MB）。
> 若下載失敗（例如網路限制），可以用瀏覽器打開腳本內列出的網址手動下載，
> 存到 `models/` 資料夾並確認檔名一致即可。

執行後會開啟一個顯示攝影機畫面的視窗，畫面下方會顯示目前偵測到的手勢文字，
方便確認辨識是否正常。偵測到特定手勢後，畫面右上角會彈出對應的 meme 圖片。

按 `ESC` 或直接關閉視窗即可結束程式。

## 專案結構

```
AnimeMeme/
  main.py                      # 進入點：開攝影機、跑主迴圈
  config.yaml                  # 攝影機編號、門檻值、冷卻時間等設定
  requirements.txt
  detectors/
    gesture_detector.py        # Hands/Pose + 11 種手勢規則分類器
  mapping/
    mapping.json               # 標籤 -> 圖片路徑對照表
    mapping_loader.py
  images/                      # 11 種手勢對應的 meme 圖片素材
  models/                      # MediaPipe 模型檔（執行 download_models.py 後會自動產生，不進版控）
  ui/
    popup_window.py            # Tkinter 彈出視窗（顯示/自動關閉）
  utils/
    stabilizer.py               # 去抖動、冷卻邏輯
    landmark_utils.py           # 距離、角度等幾何計算
    drawing.py                   # 除錯用的關鍵點繪製（純 OpenCV）
  scripts/
    download_models.py          # 下載 MediaPipe 模型檔
    action.md                   # 6 種手勢動作的原始文字描述（設計依據）
```

## 如何換成自己的 meme 圖片

1. 把想要的圖片放進 `images/` 資料夾（支援 png/jpg 等常見格式）。
2. 打開 `mapping/mapping.json`，把對應標籤的路徑改成你的圖片檔名即可，例如：

```json
{
  "pointing_finger": "images/my_pointing_meme.png"
}
```

3. 存檔後重新執行 `python main.py`（不需要改任何程式碼）。

## 如何新增新的手勢

- 修改 `detectors/gesture_detector.py`：
  - 先在 `_classify_hand_shape` 裡新增/調整單手形狀判斷（例如伸直/彎曲/張開/併攏等），
  - 再到 `GestureDetector.detect` 裡新增組合條件（可以用單手形狀、雙手相對位置、
    或搭配 Pose 的手肘/手腕/眼睛/耳朵/嘴角位置來判斷更複雜的姿勢）。
- 新增完標籤後，記得同步在 `mapping/mapping.json` 加入「標籤 -> 圖片路徑」，
  並在 `images/` 資料夾放入對應圖片。

## 調整辨識靈敏度

打開 `config.yaml`：

- `detection.min_detection_confidence` / `min_tracking_confidence`：控制 MediaPipe 偵測手/姿勢的靈敏度。
- 可以先把 `debug.show_metrics` 與 `debug.draw_landmarks` 設為 `true`，
  執行程式時觀察畫面上畫出的關鍵點與下方顯示的文字，確認手勢是否被正確分類。
- `gesture` 區塊裡的角度/距離門檻（例如手指伸直角度、握拳角度、手指張開間距）
  可以依自己的手型、鏡頭角度微調，數值越極端代表判斷越嚴格。
- `stabilizer.hold_frames`：數值越大，需要持續做同一個手勢越久才會觸發（越不容易誤判，但反應變慢）。
- `stabilizer.cooldown_seconds`：同一張 meme 圖多久內不會重複彈出。

## 提升 FPS

畫面下方會即時顯示目前的 FPS。MediaPipe 的手勢/姿勢模型是這個工具最主要的效能瓶頸，
可以從 `config.yaml` 調整：

- `detection.max_width`：偵測用畫面的寬度上限（預設 480），數值越小偵測越快，
  但太小可能讓比較遠/比較小的手不容易被偵測到。畫面顯示本身不受影響，一律用攝影機原始解析度。
- `camera.width` / `camera.height`：調低攝影機擷取解析度，能同時降低擷取、色彩轉換、
  畫面顯示的成本。
- `camera.fps`：向攝影機請求的擷取 FPS。
- 程式已經改用 `cv2.CAP_DSHOW` 後端開攝影機（Windows 預設的 MSMF 後端常常會忽略指定的
  解析度/FPS、自動幫你降級成比較保守的數值，DSHOW 通常比較準確）。

### 關於 GPU 加速與硬體上限（實測結果）

- **GPU 加速在 Windows 上目前用不了**：官方 mediapipe pip 套件的說明就寫「GPU support is
  currently limited to Ubuntu platforms」，實測在 Windows 上把 `BaseOptions` 的 `delegate`
  設成 `GPU` 會直接丟出 `NotImplementedError`（`GPU processing is disabled in build flags`），
  代表 Windows 版的 pip 套件在編譯時就沒有把 GPU 功能打進去，不是設定問題，目前無法繞過。
  若真的想用 GPU，理論上要換到 Linux（例如 WSL2 或雙系統）並自己編譯/取得有 GPU 支援的
  mediapipe 版本，但攝影機在 WSL2 裡的存取設定也相對麻煩，投入成本不低。
- **鏡頭硬體本身也有上限**：實測過這台攝影機不管用 `CAP_DSHOW` 或 `CAP_MSMF` 後端、
  不管要求 320x240 還是 640x480、不管要求 30 還是 60 FPS，實際讀取畫面的真實速度都
  穩定在約 30 FPS（`cap.get(cv2.CAP_PROP_FPS)` 有時會回報成功設定成 60，但那只是驅動
  接受了這個數值，不代表鏡頭真的能生產這麼多畫面）。也就是說**這台鏡頭實際上限大約就是
  30 FPS**，這是硬體限制，不是軟體/模型運算的問題，軟體優化能做的是盡量讓實際 FPS
  逼近這個硬體上限（原本卡在 10 幾就是因為模型運算太慢，優化後應該能跑到接近 30）。
- 如果真的需要 50~60 FPS，比較實際的做法是換一支**規格上明確標示支援 60 FPS**、
  且在裝置管理器/廠商軟體裡能選擇 60 FPS 模式的外接 USB 攝影機。

## 已知限制 / 之後可以做的事

- 手勢判斷是用手指關節角度、手掌/手指相對位置等幾何規則，精準度會因手型、光線、
  鏡頭角度而有落差，需要依個人狀況微調 `config.yaml` 的門檻值。
- `cover_one_eye`（阿嬤特拉斯）與 `josuke_pose`（東方仗助）需要同時判斷手與臉/身體
  的相對位置，是用 Pose 模型輸出的眼睛/耳朵/嘴角關鍵點做近似判斷，精準度不如專門的
  臉部網格模型，但換來不需要額外載入 `FaceLandmarker` 模型。
- `reverse_cross_palms`（反叉合掌）與 `sukuna_domain`（宿儺）都是雙手手指深度交疊、
  互相遮擋的姿勢，單一 RGB 鏡頭的 21 點手部關鍵點模型在手指重疊時準確度會明顯下降，
  目前分別用「雙手手腕貼在一起」「雙手手腕貼在一起 + 靠近嘴巴」做粗略近似，
  沒有真的檢查手指交叉/彎曲細節，穩定度比其他手勢差。
- 目前是純本機 Python 小工具；未來若要做成網頁版，可以將 `mapping.json` 與 `images/` 素材
  直接沿用，改用瀏覽器端的 MediaPipe Tasks Web (JavaScript) 重寫偵測邏輯即可。
- 若要打包成雙擊執行的 `.exe`，可以使用 `PyInstaller`（`pyinstaller main.py`），
  記得把 `config.yaml`、`mapping/`、`images/`、`models/` 等資料一併帶進打包資料夾。
- MediaPipe 在 0.10.31 版移除了舊版 `mp.solutions` API，本專案已改用新版 Tasks API；
  若之後 MediaPipe 官方 API 再有變動，需要對應調整 `detectors/` 內的程式碼。
