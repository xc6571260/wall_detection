"""
This code was coding at 2024/9/6 **Guang-Jyun, Jiang

find out nearest poi photo
"""

import os
import csv
import PIL
from PIL import Image
from math import radians, sin, cos, tan, log, dist
import math
import json
import math
import shutil

def lonlat_to_97(lon, lat):
    """
    It transforms longitude, latitude to TWD97 system.

    Parameters
    ----------
    lon : float
        longitude in degrees
    lat : float
        latitude in degrees

    Returns
    -------
    x, y [TWD97]
    """

    lat = radians(lat)
    lon = radians(lon)

    a = 6378137.0
    b = 6356752.314245
    long0 = radians(121)
    k0 = 0.9999
    dx = 250000

    e = (1 - b ** 2 / a ** 2) ** 0.5
    e2 = e ** 2 / (1 - e ** 2)
    n = (a - b) / (a + b)
    nu = a / (1 - (e ** 2) * (sin(lat) ** 2)) ** 0.5
    p = lon - long0

    A = a * (1 - n + (5 / 4.0) * (n ** 2 - n ** 3) + (81 / 64.0) * (n ** 4 - n ** 5))
    B = (3 * a * n / 2.0) * (1 - n + (7 / 8.0) * (n ** 2 - n ** 3) + (55 / 64.0) * (n ** 4 - n ** 5))
    C = (15 * a * (n ** 2) / 16.0) * (1 - n + (3 / 4.0) * (n ** 2 - n ** 3))
    D = (35 * a * (n ** 3) / 48.0) * (1 - n + (11 / 16.0) * (n ** 2 - n ** 3))
    E = (315 * a * (n ** 4) / 51.0) * (1 - n)

    S = A * lat - B * sin(2 * lat) + C * sin(4 * lat) - D * sin(6 * lat) + E * sin(8 * lat)

    K1 = S * k0
    K2 = k0 * nu * sin(2 * lat) / 4.0
    K3 = (k0 * nu * sin(lat) * (cos(lat) ** 3) / 24.0) * (
                5 - tan(lat) ** 2 + 9 * e2 * (cos(lat) ** 2) + 4 * (e2 ** 2) * (cos(lat) ** 4))

    y_97 = K1 + K2 * (p ** 2) + K3 * (p ** 4)

    K4 = k0 * nu * cos(lat)
    K5 = (k0 * nu * (cos(lat) ** 3) / 6.0) * (1 - tan(lat) ** 2 + e2 * (cos(lat) ** 2))

    x_97 = K4 * p + K5 * (p ** 3) + dx
    return x_97, y_97

def ger_cor(exif):
    lat = exif[34853][2]
    lon = exif[34853][4]
    lat = float(lat[0])+float(lat[1])/60+float(lat[2])/3600
    lon = float(lon[0])+float(lon[1])/60+float(lon[2])/3600
    lat, lon = lonlat_to_97(lon,lat)
    return lat, lon

def calculate_distance(x1, y1, x2, y2):
    """
    計算兩點之間的歐氏距離
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def poi_distribution(ori_folder, max_distance = 500):
    """
    Enter all UAV img and get the poi numbers of all photoes.

    return {filename1:1,filename2:1,filename3:2 ,....}

    """
    # 取得 poi.csv 的相對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 獲取當前執行檔案的位置
    poi_csv_path = os.path.join(script_dir, 'poi.csv')  # 使用相對路徑來獲取 CSV 檔案位置

    # 確認檔案存在
    if not os.path.exists(poi_csv_path):
        raise FileNotFoundError(f"找不到 POI CSV 檔案: {poi_csv_path}")

    poi_list = {}
    ori_files = os.listdir(ori_folder)
    location = {}
    # 讀取 POI CSV
    poi_data = []

    with open(poi_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳過表頭
        for row in reader:
            poi_id = int(row[0])
            point_x = float(row[1])
            point_y = float(row[2])
            poi_data.append((poi_id, point_x, point_y))

    for ori_file in ori_files:
        # 獲取 EXIF 資料
        ori_path = os.path.join(ori_folder, ori_file)
        pil_img = PIL.Image.open(ori_path)
        exif = pil_img._getexif()
        point_x, point_y = ger_cor(exif)
        location[ori_file] = (point_x,point_y)

        # 找到最近的 POI
        nearest_poi = None
        min_distance = float('inf')

        for poi_id, poi_x, poi_y in poi_data:
            distance = calculate_distance(point_x, point_y, poi_x, poi_y)
            if distance < min_distance:
                min_distance = distance
                nearest_poi = poi_id

        # 如果距離超過 max_distance，則不分配 POI
        if min_distance <= max_distance:
            poi_list[ori_file] = nearest_poi
        else:
            poi_list[ori_file] = None

    return poi_list, location

def poi_pick(save_folder):
    """

    """
    static_poi = {}
    # 取得 poi.csv 的相對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 獲取當前執行檔案的位置
    poi_csv_path = os.path.join(script_dir, 'poi.csv')  # 使用相對路徑來獲取 CSV 檔案位置

    # 確認檔案存在
    if not os.path.exists(poi_csv_path):
        raise FileNotFoundError(f"找不到 POI CSV 檔案: {poi_csv_path}")


    poi_data = {}
    with open(poi_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳過表頭
        for row in reader:
            poi_id = int(row[0])
            point_x = float(row[1])
            point_y = float(row[2])
            poi_data[poi_id] = (point_x, point_y)
    # Step 2: Iterate over POI folders


    poi_list = os.listdir(save_folder)

    for poi in poi_list:
        record_data_false = []
        record_data_true = []
        record_data_none = []
        final_file = None
        record_folder = os.path.join(save_folder, f"{poi}/record")
        if os.path.isdir(record_folder):
            file_list = os.listdir(record_folder)

            #Separate files based on misplacement field
            for file in file_list:
                file_path = os.path.join(record_folder, file)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    if data['misplacement'] == False and data['max_distance'] is None:
                        record_data_none.append((file, data))  # Store the data for no detection cases
                    elif data['misplacement'] == False and 'max_distance' in data:
                        record_data_false.append((file, data))  # Store the data for False cases
                    elif data['misplacement'] == True:
                        record_data_true.append((file, data))  # Store the data for True cases

            if len(record_data_true) < 2:
                max_distances = [data['max_distance'] for file, data in record_data_false if
                                 data['max_distance'] is not None]

                if len(record_data_false) > 0:
                    avg_max_distances = sum(max_distances) / len(max_distances)
                    min_distance = float('inf')

                    for file, data in record_data_false:
                        if data['max_distance'] is not None:
                            if data['max_distance'] >= avg_max_distances:
                                gps = data['gps']
                                poi_gps = poi_data[int(poi)]  # Ensure `poi` is an integer

                                # Calculate distance to POI
                                distance = calculate_distance(gps[0], gps[1], poi_gps[0], poi_gps[1])
                                if distance < min_distance:
                                    min_distance = distance
                                    final_file = file
                                    static_poi[poi] = 'green'
                else:

                    min_distance = float('inf')
                    for file, data in record_data_none:
                        gps = data['gps']
                        poi_gps = poi_data[int(poi)]

                        # Calculate distance to POI
                        distance = calculate_distance(gps[0], gps[1], poi_gps[0], poi_gps[1])
                        if distance < min_distance:
                            min_distance = distance
                            final_file = file
                            static_poi[poi] = 'gray'


                # Step 7: If more than two 'misplacement=True' files, find the closest one
            else:
                min_distance = float('inf')
                for file, data in record_data_true:
                    gps = data['gps']
                    poi_gps = poi_data[int(poi)]

                    # Calculate distance to POI
                    distance = calculate_distance(gps[0], gps[1], poi_gps[0], poi_gps[1])
                    if distance < min_distance:
                        min_distance = distance
                        final_file = file
                        static_poi[poi] = 'red'

            img_source_path = os.path.join(save_folder, f"{poi}/img/{final_file.replace('.json', '.JPG')}")
            visualize_folder = os.path.join(save_folder, f"{poi}/visualize")
            if not os.path.exists(visualize_folder):
                os.makedirs(visualize_folder)

            img_dest_path = os.path.join(visualize_folder, final_file.replace('.json', '.JPG'))
            if os.path.exists(img_source_path):
                shutil.copy(img_source_path, img_dest_path)
                print(f"Copied image {img_source_path} to {img_dest_path}")
            
    return static_poi



