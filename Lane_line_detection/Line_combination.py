"""
This code was coding at 2024/4/17 **Guang-Jyun, Jiang
"""
import numpy as np
import cv2
import math
from sympy import symbols, solve
from sklearn.cluster import KMeans
from sympy import symbols, solve
import os
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'



def k_mean(lines):
    """
    Perform k-means clustering on x values of start and end points of lines
    input:
    lines : numpy , ex: [[[x1, y1, x2, y2],...]]
    output:
    lines_clustered: list of two numpy arrays containing lines clustered by k-means ex：[[[x1, y1, x2, y2]]], [[[x1, y1, x2, y2]]]
    """
    # Initialize lists to store start and end points
    start_points = []
    if len(lines) >= 2 and lines is not None:
        for line in lines[:]:
            for x1, y1, x2, y2 in line:
                # Append start  points to the respective lists
                start_points.append([x1, y1])

        # Convert lists to numpy arrays
        start_points = np.array(start_points)

        # Perform k-means clustering with 2 clusters on x values of start and end points
        kmeans = KMeans(n_clusters=2, random_state=0).fit(start_points[:, 0].reshape(-1, 1))

        # Get the labels assigned by k-means
        labels = kmeans.labels_

        # Separate lines based on the labels
        lines_clustered = [lines[labels == i] for i in range(2)]

        # Example usage:
        # print("Cluster 1 lines:", lines_clustered[0])
        # print("Cluster 2 lines:", lines_clustered[1])

        return lines_clustered[0], lines_clustered[1]

def distance(x1, y1, x2, y2):
    """
    Calculate distance between two points (x1, y1) and (x2, y2)
    input:
    single_line: int , ex: x1, y1, x2, y2
    output:
    dist: distance between the two points
    """
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def finding_points(hough_lines, line_images, plot=False):


    def is_intersection(line1, line2, line_image, dilation_size=30, extension_length=1500):
        """
        Check if two lines intersect after dilation and extension.

        Input:
        line1: ((x1, y1), (x2, y2))
        line2: ((x1, y1), (x2, y2))
        line_image: The image on which the lines are drawn.
        dilation_size: Size of the dilation kernel.
        extension_length: Length by which the lines should be extended.

        Output:
        Returns True if the lines intersect after dilation and extension, False otherwise.
        """

        def draw_line(image, line, color, thickness=1):
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        def dilate_line(image, kernel_size):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            return cv2.dilate(image, kernel, iterations=1)

        def extend_line(line, length, image_shape):
            x1, y1,x2, y2 = line
            height, width = image_shape[:2]

            dx = x2 - x1
            dy = y2 - y1

            norm = np.sqrt(dx ** 2 + dy ** 2)
            dx = int(dx / norm * length)
            dy = int(dy / norm * length)

            x1_new, y1_new = x1 - dx, y1 - dy
            x2_new, y2_new = x2 + dx, y2 + dy

            return max(0, min(width, x1_new)), max(0, min(height, y1_new)),max(0, min(width, x2_new)), max(0, min(height, y2_new))

        # 創建影像副本來繪製和操作線條
        image1 = np.zeros_like(line_image)
        image2 = np.zeros_like(line_image)

        # 延長線條
        extended_line1 = extend_line(line1[0], extension_length, line_image.shape)
        extended_line2 = extend_line(line2[0], extension_length, line_image.shape)

        # 繪製延長後的線條
        draw_line(image1, extended_line1, 255, thickness=1)
        draw_line(image2, extended_line2, 255, thickness=1)

        # 進行膨脹操作
        dilated_image1 = dilate_line(image1, dilation_size)
        dilated_image2 = dilate_line(image2, dilation_size)
        
        combined_image = cv2.addWeighted(dilated_image1, 0.5, dilated_image2, 0.5, 0)

        # 找到重叠部分
        intersection = cv2.bitwise_and(dilated_image1, dilated_image2)

        # 检查是否存在重叠部分
        if np.any(intersection):
            return True
        else:
            return False


    def combination(line1, line2):

        """
        Combine two line of overlapping

        Input:
        line1: ((x1, y1), (x2, y2))
        line2: ((x1, y1), (x2, y2))

        Output:
        Returns the start and end points of overlapping line (BOTTOM_X,BOTTOM_Y),(TOP_X,TOP_Y)
        """
        x1, y1, x2, y2 = line1[0]
        x12, y12, x22, y22 = line2[0]
        point1, point2 = (x1, y1), (x2, y2)
        point3, point4 = (x12, y12), (x22, y22)

        # 將四個點以 y 座標進行排序
        sorted_points = sorted([point1, point2, point3, point4], key=lambda point: point[1])

        # 找出最小和最大的 y 座標點
        bottom_point = sorted_points[-1]
        top_point = sorted_points[0]

        return np.array([[bottom_point[0], bottom_point[1], top_point[0], top_point[1]]])



    def interation_intersect(all_lines,line_images):
        new_overlap_found = False
        ##remove overlap
        for i in range(len(all_lines)):
            for j in range(i + 1, len(all_lines)):
                line1 = all_lines[i]
                line2 = all_lines[j]
                if is_intersection(line1, line2,line_images):

                    all_lines = all_lines.tolist()
                    line1 = line1.tolist()
                    all_lines.remove(line1)
                    line2 = line2.tolist()
                    all_lines.remove(line2)
                    combline_line = combination(line1, line2).tolist()
                    all_lines.append(combline_line)
                    all_lines = np.array(all_lines)
                    new_overlap_found = True
                    break

            if new_overlap_found:
                break

        if new_overlap_found:
            return interation_intersect(all_lines,line_images)
        else:
            return all_lines

    def plotting_line( line_image, lines, plot):

        """
        plot the combination lines.

        Input:

        line_image: .png
        lines: numpy
        plot: bool

        Output:

        None

        """

        if plot:
          # print(f"The number of lines : {len(lines)}")
          for line in lines[:]:
            for x1, y1, x2, y2 in line:
              cv2.line(line_images, (x1, y1), (x2, y2), (255, 255, 255), 5)

    if hough_lines is not None:

        lines = interation_intersect(hough_lines,line_images)
        plotting_line( line_images, lines, plot)
        return lines

def renew_lines(lines1, lines2, height,mean_list, line_image):
    """
     redefine lines extend to edges
     input:
     lines : numpy , ex: [[[x1, y1, x2, y2],...]]
     lines : numpy , ex: [[[x1, y1, x2, y2],...]]
     mean_list: list
     output:
     new_lines1 :  numpy , ex: [[[x1, y1, x2, y2],...]]
     new_lines2 :  numpy , ex: [[[x1, y1, x2, y2],...]]
     """
    # only 1 distance
    if max(mean_list) - min(mean_list) < 50:
        points_line1 = []
        points_line2 = []
        for line in lines1[:]:
            for x1, y1, x2, y2 in line:
                points_line1.append((x1, y1))
                points_line1.append((x2, y2))
        for line in lines2[:]:
            for x1, y1, x2, y2 in line:
                points_line2.append((x1, y1))
                points_line2.append((x2, y2))
        points_line1 = sorted(points_line1, key=lambda point: point[1])
        points_line2 = sorted(points_line2, key=lambda point: point[1])
        # line equation
        #line1
        x1 ,y1, x2, y2 = points_line1[0][0], points_line1[0][1], points_line1[-1][0], points_line1[-1][1]
        if x1 - x2 == 0:
            line1 = np.array([[[x1, height, x2, 0]]])
        else:
            slope = (y2 - y1) / (x2 - x1)
            x_line1_0 = int(((0 - y1) / slope) + x1)
            x_line1_height = int(((height - y1) / slope) + x1)
            line1 = np.array([[[x_line1_height, height, x_line1_0, 0]]])
        # line2
        x1, y1, x2, y2 = points_line2[0][0], points_line2[0][1], points_line2[-1][0], points_line2[-1][1]
        if x1 - x2 == 0:
            line2 = np.array([[[x1, height, x2, 0]]])
        else:
            slope = (y2 - y1) / (x2 - x1)
            x_line2_0 = int(((0 - y1) / slope) + x1)
            x_line2_height = int(((height - y1) / slope) + x1)
            line2 = np.array([[[x_line2_height, height, x_line2_0, 0]]])

        return line1, line2

    # 2 distance happened
    else:
        if len(lines1) == 1:
            for line in lines1[:]:
                for x1, y1, x2, y2 in line:
                    if (x1 - x2) == 0:
                        lines1 = np.array([[[x1, height, x2, 0]]])
                    else:
                        slope = (y2 - y1) / (x2 - x1)
                        x_line1_0 = int(((0 - y1) / slope) + x1)
                        x_line1_height = int(((height - y1) / slope) + x1)
                        lines1 = np.array([[[x_line1_height, height, x_line1_0, 0]]])
        else:
            L1_line1, L2_line1 = k_mean(lines1)
            lines1 = []
            # Extract y1 and y2 values from each cluster
            cluster1_y = np.minimum(L1_line1[:, :, 1], L1_line1[:, :, 3])
            cluster2_y = np.minimum(L2_line1[:, :, 1], L2_line1[:, :, 3])

            # Find the minimum y value in each cluster
            min_y_cluster1 = np.min(cluster1_y)
            min_y_cluster2 = np.min(cluster2_y)

            if min_y_cluster1 < min_y_cluster2:
                points_line1 = []
                points_line2 = []
                for line in L1_line1[:]:
                    for x1, y1, x2, y2 in line:
                        points_line1.append((x1, y1))
                        points_line1.append((x2, y2))
                for line in L2_line1 [:]:
                    for x1, y1, x2, y2 in line:
                        points_line2.append((x1, y1))
                        points_line2.append((x2, y2))
                points_line1 = sorted(points_line1, key=lambda point: point[1])
                points_line2 = sorted(points_line2, key=lambda point: point[1])
                # line equation
                # line1
                x1_l1, y1_l1, x2_l1, y2_l1 = points_line1[0][0], points_line1[0][1], points_line1[-1][0], points_line1[-1][1]
                x1_l2, y1_l2, x2_l2, y2_l2 = points_line2[0][0], points_line2[0][1], points_line2[-1][0], points_line2[-1][1]
                if x1_l1 - x2_l1 == 0:
                    lines1.append([[x2_l1, y2_l1, x1_l1, 0]])
                else:
                    slope = (y2_l1 - y1_l1) / (x2_l1 - x1_l1)
                    x_line1_0 = int(((0 - y1_l1) / slope) + x1_l1)
                    lines1.append([[x2_l1, y2_l1, x_line1_0, 0]])
                # line2
                if x1_l2 - x2_l2 == 0:
                    lines1.append([[x2_l2, height, x1_l2, y1_l2]])
                else:
                    slope = (y2_l2 - y1_l2) / (x2_l2 - x1_l2)
                    x_line2_height = int(((height - y1_l2) / slope) + x1_l2)
                    lines1.append([[x_line2_height, height, x1_l2, y1_l2]])
                lines1 = np.array(lines1)
                # plot box
                cv2.rectangle(line_image, (x2_l1-100, y2_l1-100), (x1_l2+100, y1_l2+100), (0, 0, 255), 5)
            else:
                points_line1 = []
                points_line2 = []
                for line in L1_line1[:]:
                    for x1, y1, x2, y2 in line:
                        points_line1.append((x1, y1))
                        points_line1.append((x2, y2))
                for line in L2_line1[:]:
                    for x1, y1, x2, y2 in line:
                        points_line2.append((x1, y1))
                        points_line2.append((x2, y2))
                points_line1 = sorted(points_line1, key=lambda point: point[1])
                points_line2 = sorted(points_line2, key=lambda point: point[1])
                # line equation
                # line1
                x1_l1, y1_l1, x2_l1, y2_l1 = points_line1[0][0], points_line1[0][1], points_line1[-1][0], points_line1[-1][1]
                x1_l2, y1_l2, x2_l2, y2_l2 = points_line2[0][0], points_line2[0][1], points_line2[-1][0], points_line2[-1][1]
                if x1_l1 - x2_l1 == 0:
                    lines1.append([[x2_l1, height, x1_l1, y1_l1]])
                else:
                    slope = (y2_l1 - y1_l1) / (x2_l1 - x1_l1)
                    x_line1_height = int(((height - y1_l1) / slope) + x1_l1)
                    lines1.append([[x_line1_height, height, x1_l1, y1_l1]])
                # line2

                if x1_l2 - x2_l2 == 0:
                    lines1.append([[x2_l2, y2_l2, x1_l2, 0]])
                else:
                    slope = (y2_l2 - y1_l2) / (x2_l2 - x1_l2)
                    x_line2_0 = int(((0 - y1_l2) / slope) + x1_l2)
                    lines1.append([[x2_l2, y2_l2, x_line2_0, 0]])
                lines1 = np.array(lines1)
                # plot box
                cv2.rectangle(line_image, (x2_l2 - 100, y2_l2 - 100), (x1_l1 + 100, y1_l1 + 100), (255, 255, 255), 5)

        if len(lines2) == 1:
            for line in lines2[:]:
                for x1, y1, x2, y2 in line:
                    if (x1 - x2) == 0:
                        lines2 = np.array([[[x1, height, x2, 0]]])
                    else:
                        slope = (y2 - y1) / (x2 - x1)
                        x_line2_0 = int(((0 - y1) / slope) + x1)
                        x_line2_height = int(((height - y1) / slope) + x1)
                        lines2 = np.array([[[x_line2_height, height, x_line2_0, 0]]])
        else:
            L1_line2, L2_line2 = k_mean(lines2)
            lines2 = []
            # Extract y1 and y2 values from each cluster
            cluster1_y = np.minimum(L1_line2[:, :, 1], L1_line2[:, :, 3])
            cluster2_y = np.minimum(L2_line2[:, :, 1], L2_line2[:, :, 3])

            # Find the minimum y value in each cluster
            min_y_cluster1 = np.min(cluster1_y)
            min_y_cluster2 = np.min(cluster2_y)

            if min_y_cluster1 < min_y_cluster2:
                points_line1 = []
                points_line2 = []
                for line in L1_line2[:]:
                    for x1, y1, x2, y2 in line:
                        points_line1.append((x1, y1))
                        points_line1.append((x2, y2))
                for line in L2_line2[:]:
                    for x1, y1, x2, y2 in line:
                        points_line2.append((x1, y1))
                        points_line2.append((x2, y2))
                points_line1 = sorted(points_line1, key=lambda point: point[1])
                points_line2 = sorted(points_line2, key=lambda point: point[1])
                # line equation
                # line1
                x1_l1, y1_l1, x2_l1, y2_l1 = points_line1[0][0], points_line1[0][1], points_line1[-1][0], points_line1[-1][1]
                x1_l2, y1_l2, x2_l2, y2_l2 = points_line2[0][0], points_line2[0][1], points_line2[-1][0], points_line2[-1][1]
                if x1_l1 - x2_l1 == 0:
                    lines2.append([[x2_l1, y2_l1, x1_l1, 0]])
                else:
                    slope = (y2_l1 - y1_l1) / (x2_l1 - x1_l1)
                    x_line1_0 = int(((0 - y1) / slope) + x1_l1)
                    lines2.append([[x2_l1, y2_l1, x_line1_0, 0]])
                # line2

                if x1_l2 - x2_l2 == 0:
                    lines2.append([[x2_l2, height, x1_l2, y1_l2]])
                else:
                    slope = (y2_l2 - y1_l2) / (x2_l2 - x1_l2)
                    x_line2_height = int(((height - y1_l2) / slope) + x1_l2)
                    lines2.append([[x_line2_height, height, x1_l2, y1_l2]])
                lines2 = np.array(lines2)
                # plot box
                cv2.rectangle(line_image, (x2_l1 - 10, y2_l1 - 10), (x1_l2 + 10, y1_l2 + 10), (255, 255, 255), 5)
            else:
                points_line1 = []
                points_line2 = []
                for line in L1_line2[:]:
                    for x1, y1, x2, y2 in line:
                        points_line1.append((x1, y1))
                        points_line1.append((x2, y2))
                for line in L2_line2[:]:
                    for x1, y1, x2, y2 in line:
                        points_line2.append((x1, y1))
                        points_line2.append((x2, y2))
                points_line1 = sorted(points_line1, key=lambda point: point[1])
                points_line2 = sorted(points_line2, key=lambda point: point[1])
                # line equation
                # line1
                x1_l1, y1_l1, x2_l1, y2_l1 = points_line1[0][0], points_line1[0][1], points_line1[-1][0], points_line1[-1][1]
                x1_l2, y1_l2, x2_l2, y2_l2 = points_line2[0][0], points_line2[0][1], points_line2[-1][0], points_line2[-1][1]
                if x1_l1 - x2_l1 == 0:
                    lines2.append([[x2_l1, height, x1_l1, y1_l1]])
                else:
                    slope = (y2_l1 - y1_l1) / (x2_l1 - x1_l1)
                    x_line1_height = int(((height - y1_l1) / slope) + x1_l1)
                    lines2.append([[x_line1_height, height, x1_l1, y1_l1]])
                # line2
                if x1_l2 - x2_l2 == 0:
                    lines2.append([[x1_l2, y1_l2, x2_l2, 0]])
                else:
                    slope = (y2_l2 - y1_l2) / (x2_l2 - x1_l2)
                    x_line2_0 = int(((0 - y1_l2) / slope) + x1_l2)
                    lines2.append([[x2_l2, y2_l2, x_line2_0, 0]])
                lines2 = np.array(lines2)
                # plot box
                cv2.rectangle(line_image, (x2_l2 - 10, y2_l2 - 10), (x1_l1 + 10, y1_l1 + 10), (255, 255, 255), 5)
                
        return lines1, lines2


def outlier_eliminate(lines, line_image):
    """
    eliminate over three lines in the y axis, keep two lines in the one y axis
    input:
    lines : numpy , ex: [[[x1, y1, x2, y2],...]]
    output:
    lines : numpy , ex: [[[x1, y1, x2, y2],...]]
    """
    height = line_image.shape[1]

    def calculate_intersection(line, y):
        """
        Calculate the intersection of a line with a horizontal line at y.
        """
        x1, y1, x2, y2 = line[0]

        if y1 == y2:  # Horizontal line, so intersection is the whole segment
            return min(x1, x2), max(x1, x2)

        # Linear interpolation to find the intersection x coordinate
        x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        # GSD=0.67
        return x * 0.67

    def slope_filter(lines, slope):
        """
        If line is not vertical, delete it.
        """
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 != x2:
                m = (x1 - x2) / (y1 - y2)
                if abs(m) <= slope:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
                
        return np.array(filtered_lines)




    lines = slope_filter(lines, 0.1)

    lines_to_keep = []
    for y in range(height + 1):
        # 计算水平线 y = y 与所有线段的交点
        intersections = {}
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y2 <= y <= y1 or y1 <= y <= y2:  # 确保处理线段的方向
                x = int(calculate_intersection(line, y))
                intersections[x] = line

        # 计算交点之间的距离
        distances = []
        intersection_points = list(intersections.keys())
        if len(intersection_points) >= 2:
            for i in range(len(intersection_points)):
                for j in range(i + 1, len(intersection_points)):
                    distance = abs(intersection_points[i] - intersection_points[j])
                    distances.append((distance, intersection_points[i], intersection_points[j]))


            # 仅保留距离最接近 300 的两个线段
            remaining_indices = []
            closest_dist = None
            # print(distances)
            for dist, x1, x2 in distances:
                # print(dist,intersections[x1],intersections[x2])
                if 150 <= dist <= 330:  # 根据需要调整阈值
                    # 如果当前距离更接近 300，更新最近距离和线段索引
                    if closest_dist is None or abs(dist - 300) < abs(closest_dist - 300):
                        closest_dist = dist
                        remaining_indices = [x1, x2]

                # 确保 lines_to_keep 中在当前 y 上仅保留两条线段
            if len(remaining_indices) == 2:
                for x in remaining_indices:
                    line_to_add = intersections[x]
                    if len(lines_to_keep) < 2 or not any(np.array_equal(line_to_add, existing_line) for existing_line in lines_to_keep):
                        lines_to_keep.append(line_to_add)

    # Convert lines_to_keep back to a NumPy array
    lines = np.array(lines_to_keep)


                    
    return lines




def cal_distance(lines, line_image, plot = True):
    """
    calaulate maximum and minimun distance between walls
    input:
    lines : numpy , ex: [[[x1, y1, x2, y2],...]]
    output:
    max_distance: int
    min_distance: int
    plot: bool
    """
    width, height = line_image.shape[0],  line_image.shape[1]

    def local_cal_points(line1, line2):
        """
        Calculate distance between two line of the walls
        input:
        line1: numpy , ex: [[x1, y1, x2, y2]]
        line2: numpy , ex: [[x1, y1, x2, y2]]
        output:
        return:(points on long_line, points on short_line) :list
        """

        def check_line_length(line1,line2):
            """
            Check which line is the longest
            input:
            line1: numpy , ex: [[x1, y1, x2, y2]]
            line2: numpy , ex: [[x1, y1, x2, y2]]
            output:
            return: index, 0 or 1
            """
            x1, y1, x2, y2 = line1[0]
            x12, y12, x22, y22 = line2[0]
            if distance(x1, y1, x2, y2) > distance(x12, y12, x22, y22):
                return True
            else:
                return False

        def slope(line):
            """
            Calculate the slope of line
            input:
            line1: numpy , ex: [[x1, y1, x2, y2]]
            output:
            return: int
            """
            x1, y1, x2, y2 = line[0]
            if x1 - x2 == 0:
                return None
            else:
                return ( y2 - y1 ) / ( x2 - x1 )

        def find_all_distance(long_line, short_line):
            """
            find the all points per pixel on short line to long_line
            input:
            long_line: numpy , ex: [[x1, y1, x2, y2]]
            short_line: numpy , ex: [[x1, y1, x2, y2]]
            output:
            return:(points on long_line, points on short_line) :list
            """

            # To find all the points on the short line
            points_short = []
            x12, y12, x22, y22 = short_line[0]
            slope2 = slope(short_line)
            if slope2 is not None and abs(x12-x22) >= 20 :
                for x_short in range(min(x12, x22), max(x12, x22) + 1):
                    x = x_short
                    y_short = y12 + slope2 * (x - x12)
                    if y_short <= height:
                        points_short.append((x, y_short))
            else:  # vertical line
                for y_short in range(min(y12, y22), max(y12, y22) + 1):
                    points_short.append((x12, y_short))

            points_long = []
            x1, y1, x2, y2 = long_line[0]
            slope1 = slope(long_line)

            remove_indices = []  # List to store indices of points to remove from points_short

            if slope1 is not None:
                for i, (x_sh, y_sh) in enumerate(points_short):
                    if min(y1, y2) <= y_sh <= max(y1, y2):
                        x_long = ((y_sh - y1) / slope1) + x1
                        points_long.append((x_long, y_sh))
                    else:
                        remove_indices.append(i)
            else:
                for y_short in range(min(y12, y22), max(y12, y22) + 1):
                    points_long.append((x1, y_short))

            # Remove points from points_short
            for index in reversed(remove_indices):
                del points_short[index]

            # Convert points to integer coordinates
            points_long = [(int(x), int(y)) for x, y in points_long]
            points_short = [(int(x), int(y)) for x, y in points_short]

            return points_long, points_short


        if  check_line_length(line1,line2):
            return find_all_distance(line1, line2)
        else:
            return find_all_distance(line2, line1)

    def statics_dis(points_long, points_short):
        """
        find the all points per pixel on short line to long_line
        input:
        points_long: list , ex: [(x1, y1), (x2,y2), ..]
        points_short: list , ex: [(x1, y1), (x2,y2), ..]
        output:
        return:max_dis, min_dis, mean_dis
        """
        distance_local = []
        for i in range(len(points_short)):
            x1, y1 = points_long[i]
            x2, y2 = points_short[i]
            distance_local.append(distance(x1, y1, x2, y2))
        return max(distance_local), min(distance_local), sum(distance_local)/len(distance_local)


    record = {}
    if lines is not  None:
        if 4 > len(lines) >= 2:
            max_distance = []
            min_distance = []
            mean_distance = []
            lines1, lines2 = k_mean(lines)
            start_plot = []
            end_plot = []
            for line1 in lines1:
                for line2 in lines2:
                    start_points, end_points = local_cal_points(line1, line2)
                    if start_points and end_points :
                        start_plot.append(start_points)
                        end_plot.append(end_points)
                        max_dis, min_dis, mean_dis = statics_dis(start_points, end_points)
                        max_distance.append(max_dis)
                        min_distance.append(min_dis)
                        mean_distance.append(mean_dis)


            # plot distance
            if max_distance:

                # 如果距离小于100，跳到指定部分
                if max(max_distance) < 100:
                    distance_text = (f"No detection")
                    cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (0, 0, 255), 10)
                    record['max_distance'] = None
                    record['min_distance'] = None
                    record['misplacement'] = False
                    return record

                # 计算最大和最小距离，并将结果转换为整数
                max_dist = int(max(max_distance) * 0.58)
                min_dist = int(min(min_distance) * 0.58)

                if (max(max_distance) - min(min_distance)) < 50:
                    distance_text = (f"MAX_DISTANCE = {max_dist} cm "
                                     f" MIN_DISTANCE = {min_dist} cm")

                    cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (255, 255, 255), 10)
                    record['max_distance'] = max_dist
                    record['min_distance'] = min_dist
                    record['misplacement'] = False
                else:
                    distance_text = (f"MAX_DISTANCE = {max_dist} cm "
                                     f" MIN_DISTANCE = {min_dist} cm "
                                     f"offset detection")
                    record['max_distance'] = max_dist
                    record['min_distance'] = min_dist
                    record['misplacement'] = False
                    cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (0, 0, 255), 10)

                # update new_line
                max_new_distance = []
                min_new_distance = []
                mean_new_distance = []
                start_plot = []
                end_plot = []
                lines1_new, lines2_new = renew_lines(lines1, lines2, height,  mean_distance, line_image)
                for lines in lines1_new:
                    x1, y1,x2, y2 = lines[0]
                    # print(x1, y1,x2, y2)
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
                for lines in lines2_new:
                    x1, y1, x2, y2 = lines[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
                for line1 in lines1_new:
                    for line2 in lines2_new:
                        start_points, end_points = local_cal_points(line1, line2)
                        if start_points and end_points:
                            start_plot.append(start_points)
                            end_plot.append(end_points)
                            max_dis, min_dis, mean_dis = statics_dis(start_points, end_points)
                            max_new_distance.append(max_dis)
                            min_new_distance.append(min_dis)
                            mean_new_distance.append(mean_dis)

                for i in range(min(len(start_plot),len(end_plot))):
                    for j in range(min(len(start_plot[i]),len(end_plot[i]))):

                        if (max(max_new_distance) - mean_new_distance[i]) < 50 or len(lines1_new)+len(lines2_new) == 2:
                            cv2.line(line_image, start_plot[i][j], end_plot[i][j], (0, 255, 0), 5)
                        else:
                            cv2.line(line_image, start_plot[i][j], end_plot[i][j], (255, 0, 0), 5)

        elif len(lines) == 4:
            max_distance = []
            min_distance = []
            mean_distance = []
            lines1, lines2 = k_mean(lines)
            start_plot = []
            end_plot = []
            for line1 in lines1:
                for line2 in lines2:
                    start_points, end_points = local_cal_points(line1, line2)
                    if start_points and end_points:
                        start_plot.append(start_points)
                        end_plot.append(end_points)
                        max_dis, min_dis, mean_dis = statics_dis(start_points, end_points)
                        max_distance.append(max_dis)
                        min_distance.append(min_dis)
                        mean_distance.append(mean_dis)
            # plot distance
            if max_distance:

                # 计算最大和最小距离，并将结果转换为整数
                max_dist = int(max(max_distance) * 0.58)
                min_dist = int(min(min_distance) * 0.58)

                # 创建一个列表来存储所有的点 (x, y)
                points = []

                # 遍历 lines，将每条线的两个端点添加到 points 列表
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    points.append((x1, y1))
                    points.append((x2, y2))

                # 按 y 坐标对 points 进行排序，如果 y 坐标相同，则按 x 坐标排序
                sorted_points = sorted(points, key=lambda p: (p[1], p[0]))

                # 选择排序后的第3个到第6个点
                selected_points = sorted_points[2:6]

                # 计算平均坐标
                avg_x = int(np.mean([p[0] for p in selected_points]))
                avg_y = int(np.mean([p[1] for p in selected_points]))
                avg_point = (avg_x, avg_y)
                # 定义正方形的边长
                side_length = 1000

                # 计算正方形的四个顶点
                half_side = side_length // 2
                square_points = [
                    (avg_x - half_side, avg_y - half_side),
                    (avg_x + half_side, avg_y - half_side),
                    (avg_x + half_side, avg_y + half_side),
                    (avg_x - half_side, avg_y + half_side)
                ]

                # 在图像上绘制正方形
                cv2.polylines(line_image, [np.array(square_points)], isClosed=True, color=(0, 0, 255), thickness=10)
                distance_text = (f"MAX_DISTANCE = {max_dist} cm "
                                 f" MIN_DISTANCE = {min_dist} cm "
                                 f"misplacement detection!!!")

                record['max_distance'] = max_dist
                record['min_distance'] = min_dist
                record['misplacement'] = True
                cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 0, 255), 10)

        else:
            distance_text = (f"No detection")
            record['max_distance'] = None
            record['min_distance'] = None
            record['misplacement'] = False

            cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 0, 255), 10)
    else:
        distance_text = (f"No detection")
        record['max_distance'] = None
        record['min_distance'] = None
        record['misplacement'] = False

        cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 0, 255), 10)

    return record

            