"""
This code was coding at 2024/4/17 **Guang-Jyun, Jiang
"""

import cv2
import os
import numpy as np
from Lane_line_detection.Line_combination import finding_points
from Lane_line_detection.Line_combination import cal_distance
from Lane_line_detection.Line_combination import outlier_eliminate
from poi_pick.poi import poi_distribution
from poi_pick.poi import poi_pick
from skimage import morphology
from model.predict import generate_mask
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import img_as_ubyte
from PIL import Image
import json




class LinePredictor:
    
    def __init__(self,model, plot=True):
        self.plot = plot
        self.model = model


    def preprocessing(self, mask_img, ori_img):
        # 读取原始图像
        image = cv2.imread(ori_img)

        # 将掩码图像转换为 RGB 格式
        mask_img = np.clip(mask_img, 0, 255).astype(np.uint8)
        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        mask= cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        return mask


    def hough_transform(self, edges_img):

        rho = 1
        theta = np.pi / 180
        threshold = 1
        min_line_length = 300
        max_line_gap = 200

        return cv2.HoughLinesP(edges_img, rho, theta, threshold, np.array([]),
                               min_line_length, max_line_gap)


    def plot_line(self, hough_lines, line_image, plot_img_line):

        if hough_lines is not None:
            if plot_img_line:
                for line in hough_lines[:]:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)


    def predict_line(self, mask_img, ori_img):
        image = cv2.imread(ori_img)
        line_image = np.zeros_like(image)

        edges = self.preprocessing(mask_img, ori_img)

        lines = self.hough_transform(edges)
        self.plot_line(lines, line_image, False)

        lines = finding_points(lines, line_image, True)

        if lines is not None:
            lines = outlier_eliminate(lines, line_image)

        record = cal_distance(lines, line_image,self.plot)
        result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

        return result,record


    def predict_lines_for_folder(self, ori_folder, save_folder):

        poi , gps_point = poi_distribution(ori_folder)
        ori_files = os.listdir(ori_folder)

        for ori_file in ori_files:
            ori_img_path = os.path.join(ori_folder, ori_file)

            if ori_file.endswith('.JPG') or ori_file.endswith('.jpg'):

                # Generate mask and prediction result
                mask_img = generate_mask(ori_img_path, self.model)

                result_image, record_txt = self.predict_line(mask_img, ori_img_path)
                try:
                    record_txt["gps"] = gps_point[ori_file]
                except:
                    record_txt["gps"] = None

                # Get corresponding POI value
                poi_value = poi.get(ori_file)  # Get POI value based on image name

                # Set folder path based on POI value
                if poi_value is None:
                    poi_folder = os.path.join(save_folder, 'None')  # Folder for images with no POI
                else:
                    poi_folder = os.path.join(save_folder, str(poi_value))  # POI folder

                # Create POI folder if it doesn't exist
                img_folder = os.path.join(poi_folder, 'img')  # Image folder
                record_folder = os.path.join(poi_folder, 'record')  # Record folder

                # Create folders if they don't exist
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder)
                if not os.path.exists(record_folder):
                    os.makedirs(record_folder)

                # Save result image in i/img/filename
                result_img_path = os.path.join(img_folder, ori_file)
                print(f"Saving {ori_file} image to {img_folder} ...")
                cv2.imwrite(result_img_path, result_image)

                # Save record_txt as JSON in i/record/filename.json
                record_json_path = os.path.join(record_folder, f"{os.path.splitext(ori_file)[0]}.json")
                print(f"Saving {ori_file} record to {record_folder} as JSON ...")
                with open(record_json_path, 'w') as file:
                    json.dump(record_txt, file, indent=4)  # Save as JSON format with indentation

        #Find the final photo
        static_poi = poi_pick(save_folder)
        result_folder = os.path.join(save_folder,"result")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        result_file_path = os.path.join(save_folder, "result", "results.json")
        with open(result_file_path, 'w') as json_file:
            json.dump(static_poi, json_file, indent=4)
