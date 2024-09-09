"""
This code was coding at 2024/4/17 **Guang-Jyun, Jiang
"""
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from Line_predict.hough_transform import LinePredictor
import os


current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "YOLOv8_model", "best.pt")




def main():
    """
    功能:
    這個函數會從指定的資料夾中讀取 POI (Point of Interest) 資料和相關的記錄檔案，並根據條件篩選出與 POI 最近的 UAV (無人機) 影像，將篩選出的影像存入指定資料夾中。

    資料夾結構:
    ori_folder:
        - UAV img folder: 無人機拍攝的影像檔案資料夾。
            結構:
            ├── ori_folder/
            │   ├── img1.JPG
            │   ├── img2.JPG

    save_folder:
        - 這個資料夾將儲存 POI 篩選後對應的影像檔案。
            結構:
            ├── save_folder/
            │   ├── POI1/
            │   │   ├── img/
            │   │   │   ├── img1.JPG
            │   │   │   ├── img2.JPG
            │   │   └── record/
            │   │       ├── record1.json
            │   │       ├── record2.json
            │   │   └── visualize/
            │   │       ├── img1.JPG
            │   ├── POI2/
            │   │   ├── img/
            │   │   │   ├── img3.JPG
            │   │   │   ├── img4.JPG
            │   │   └── record/
            │   │       ├── img3.json
            │   │       ├── img4.json
            │   │   └── visualize/
            │   │       ├── img3.JPG
            │   └── POI3/

    參數:
    - save_folder: 儲存篩選後影像的資料夾路徑。

    程式流程:
    1. 讀取 poi.csv 檔案中的 POI 座標資料。
    2. 遍歷 ori_folder 中的每一個 POI 資料夾，並經過YOLOv8預測與HOUGH直線預測結果輸出影像
    3. 如果 misplacement=True 的檔案數量少於 2，則根據 max_distance 的平均值篩選出與 POI 距離最近的影像。
    4. 如果 misplacement=True 的檔案數量超過 2，則直接找出距離 POI 最近的影像。
    5. 最終，將篩選出的影像複製到 save_folder 中對應的 POI 資料夾下的 visualize 子資料夾中。
    6. 輸出為dict顯示預測完每個POI的結果，綠色為檢測通過，灰色為無檢測到物件，紅色為異常 {1:green,2:red,3:gray....}
    """

    ori_folder = r"D:\champion\NTU_project\港研\資料集\20240429資料集\胸牆"
    save_folder = r"D:\champion\NTU_project\港研\YOLOv8\YOLOv8_train\wall_detection\poi_5m"

    if not os.path.exists(save_folder):
        # If it doesn't exist, create it
        os.makedirs(save_folder)

    model = YOLO(model_path)
    line_predictor = LinePredictor(model, plot=True)
    line_predictor.predict_lines_for_folder(ori_folder, save_folder)




if __name__ == '__main__':
    main()