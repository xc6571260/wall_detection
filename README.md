# UAV 影像處理與 PCI 篩選系統

## 專案簡介

此專案針對無人機 (UAV) 拍攝的影像進行線條檢測、物件偵測與 POI (Point of Interest) 分析。系統會從指定的資料夾中讀取 PCI 資料與相關的記錄檔案，並依據條件篩選出與 PCI 最近的影像，將篩選結果存入指定資料夾，並且輸出預測結果狀態的摘要。

## 目標與功能

- 讀取無人機拍攝的影像進行 YOLOv8 物件檢測和 Hough 直線預測。
- 根據 PCI 資料 (如 GPS 座標)，篩選出最接近的影像。
- 根據影像是否出現異常 (misplacement)，對結果進行篩選，並分類異常與正常的 PCI。
- 自動化結果儲存，並將篩選出的影像儲存在對應資料夾中。
- 最終輸出結果為包含每個 PCI 檢測狀態的字典：
  - 綠色 (green): 檢測通過
  - 紅色 (red): 異常
  - 灰色 (gray): 無檢測到物件


## 資料夾說明

- **ori_folder**: 儲存原始 UAV 影像的資料夾，影像檔案將在此資料夾中讀取。
  - **示例檔案**: `img1.JPG`, `img2.JPG`

- **save_folder**: 儲存處理後結果的資料夾，依據 POI 分類。
  - **POI1/**: 儲存 POI1 相關資料的資料夾。
    - **img/**: 儲存 POI1 相關影像檔案的資料夾。
      - **示例檔案**: `img1.JPG`, `img2.JPG`
    - **record/**: 儲存 POI1 相關記錄檔案的資料夾（JSON 格式）。
      - **示例檔案**: `img1.json`, `img2.json`
    - **visualize/**: 儲存篩選後影像的資料夾，用於可視化結果。
      - **示例檔案**: `img1.JPG`
  - **POI2/**: 儲存 POI2 相關資料的資料夾。
    - **img/**: 儲存 POI2 相關影像檔案的資料夾。
      - **示例檔案**: `img3.JPG`, `img4.JPG`
    - **record/**: 儲存 POI2 相關記錄檔案的資料夾（JSON 格式）。
      - **示例檔案**: `img3.json`, `img4.json`
    - **visualize/**: 儲存篩選後影像的資料夾，用於可視化結果。
      - **示例檔案**: `img3.JPG`

## 路徑範例

- **原始影像資料夾路徑**:
  - `ori_folder/img1.JPG`
  - `ori_folder/img2.JPG`

- **篩選後影像儲存路徑**:
  - `save_folder/POI1/visualize/img1.JPG`
  - `save_folder/POI2/visualize/img3.JPG`

## 程式說明

### 輸入

- `ori_folder`：包含 UAV 拍攝的原始影像資料夾。

### 輸出

1. 篩選後影像：儲存至 `save_folder` 中的 `visualize` 資料夾。
2. 檢測結果摘要：篩選結果會輸出為一個字典，顯示每個 PCI 的檢測狀態：
   - green: 檢測通過
   - red: 出現異常
   - gray: 無檢測到物件

### 流程說明

1. 讀取 `poi.csv` 中的 PCI 座標資料。
2. 遍歷 `ori_folder` 中的每個 POI 資料夾，進行 YOLOv8 預測與 Hough 直線檢測。
3. 根據 `misplacement` 狀態 (True/False) 來篩選影像：
   - 若 `misplacement=True` 檔案數量少於 2，則根據 `max_distance` 平均值來選擇最近的影像。
   - 若 `misplacement=True` 檔案數量大於 2，則直接選擇距離 POI 最近的影像。
4. 將篩選出的影像存入對應的 PCI 資料夾內的 `visualize` 子資料夾中。
5. 最終輸出字典格式的預測結果狀態。

## 安裝與使用

### 環境需求

- Python 3.x
- 必要套件：
  - `os`
  - `csv`
  - `json`
  - `shutil`
  - 其他 YOLOv8 與 Hough 直線偵測相關套件

### 執行方式

1. 將無人機影像資料放入 `ori_folder` 中。
2. 執行主程式，系統將會進行 PCI 影像篩選與預測，並將結果存入 `save_folder` 中。


