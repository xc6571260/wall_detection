"""
This code was coding at 2024/4/17 **Guang-Jyun, Jiang
"""
import numpy as np
import cv2
import math
from sympy import symbols, solve
from sklearn.cluster import KMeans
from sympy import symbols, solve


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

    def detection_region(start_point, end_point, offset=50):
        """
        This function is used to plot a detection box around a single line.

        input:
        start_point: (x1, y1) : tuple
        end_point: (x2, y2) : tuple
        offset: int
  
        output:
        return (R_down_X, R_down_y), (L_up_x, L_up_y)
        """
        return (start_point[0] + offset, start_point[1]), (start_point[0] - offset, end_point[1])


    def is_overlap(line1, line2):

        """
        Check if two rectangles overlap.

        Input:
        line1: ((x1, y1), (x2, y2))
        line2: ((x1, y1), (x2, y2))

        Output:
        Returns True if the rectangles overlap, False otherwise.
        """
        if line1[0][1] > line1[0][3]:
            x1, y1, x2, y2 = line1[0]
        else:
            x2, y2, x1, y1 = line1[0]

        if line2[0][1] > line2[0][3]:
            x12, y12, x22, y22 = line2[0]
        else:
            x22, y22, x12, y12 = line2[0]

        # Unpack coordinates of both rectangles
        (x1_rect1, y1_rect1), (x2_rect1, y2_rect1) = detection_region((x1, y1), (x2, y2))
        (x1_rect2, y1_rect2), (x2_rect2, y2_rect2) = detection_region((x12, y12), (x22, y22))

        if (x1_rect1 < x2_rect2 or
                x1_rect2 < x2_rect1 or
                y1_rect1 < y2_rect2 or
                y1_rect2 < y2_rect1):

            return False
        else:
            return True



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

    def interation_lines(all_lines):
        """
        finding all the overlap line in all the lines.

        Input:
        lines: [line1, line2, line3 ...]
        Output:
        Returns the list of the overlapping line: [line1, line2, line3]
        """
        new_overlap_found = False

        for i in range(len(all_lines)):
            for j in range(i + 1, len(all_lines)):
                line1 = all_lines[i]
                line2 = all_lines[j]
                if is_overlap(line1, line2):

                    if not np.any(np.array_equal(combination(line1, line2), line1)):
                        all_lines = all_lines.tolist()
                        line1 = line1.tolist()
                        all_lines.remove(line1)
                        all_lines = np.array(all_lines)

                    if not np.any(np.array_equal(combination(line1, line2), line2)):
                        all_lines = all_lines.tolist()
                        line2 = line2.tolist()
                        all_lines.remove(line2)
                        all_lines = np.array(all_lines)
                        
                    if all_lines is not None and all_lines.size > 0:
                        if not np.any(np.all(all_lines == combination(line1, line2), axis=1)):
                            all_lines = all_lines.tolist()
                            combline_line = combination(line1, line2).tolist()
                            all_lines.append(combline_line)
                            all_lines = np.array(all_lines)
                    new_overlap_found = True

                    break

            if new_overlap_found:
                break

        if new_overlap_found:
            return interation_lines(all_lines)
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
          print(f"The number of lines : {len(lines)}")
          for line in lines[:]:
            for x1, y1, x2, y2 in line:
              cv2.line(line_images, (x1, y1), (x2, y2), (255, 255, 255), 5)

    def data_clean(lines, tolerate = 50):
        """
        delete oulier
        iuput:
        lines : numpy
        output:
        lines : numpy
        """
        # remove too short line
        if lines is not None:
            lines = lines.tolist()
            for line in lines[:]:
                x1, y1, x2, y2 = line[0]
                if abs(y2-y1) < 100:
                    lines.remove(line)
            lines = np.array(lines)

        if len(lines) > 2 and lines is not None:
            dis = []
            lines = lines.tolist()
            for line in lines[:]:
                x1, y1, x2, y2 = line[0]
                dis.append(distance(x1, y1, x2, y2))


            # 找出前二高的索引
            sorted_indices = sorted(range(len(dis)), key=lambda i: dis[i], reverse=True)
            top_two_indices = sorted_indices[:2]

            x_max = None
            x_min = None
            # 獲得前二高距離對應的兩條直線的端點 (x1, x2) 的值
            top_two_lines = [lines[i][0] for i in top_two_indices]
            for line in top_two_lines:
                x1, y1, x2, y2 = line
                if x_max is None:
                    x_max = max(x1,x2)
                    x_min = min(x1,x2)
                else:
                    x_max = max(x_max,x1,x2)
                    x_min = min(x_min,x1,x2)

            for line in lines[:]:
                x1, y1, x2, y2 = line[0]
                if max(x1,x2) > x_max + tolerate or min(x1,x2) < x_min - tolerate:

                    # print("remove", line)
                    lines.remove(line)

            lines = np.array(lines)
            return lines
        else:
            return lines

    if hough_lines is not None:
      lines = interation_lines(hough_lines)
      lines = data_clean(lines)
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
                cv2.rectangle(line_image, (x2_l1-10, y2_l1-10), (x1_l2+10, y1_l2+10), (255, 255, 255), 5)
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
                cv2.rectangle(line_image, (x2_l2 - 10, y2_l2 - 10), (x1_l1 + 10, y1_l1 + 10), (255, 255, 255), 5)

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



    if lines is not  None:
        if len(lines) >= 2:
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
                if (max(max_distance) - min(min_distance)) < 50:
                    distance_text = (f"MAX_DISTANCE = {max(max_distance)} pixel "
                                     f" MIN_DISTANCE = {min(min_distance)} pixel")

                    cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (255, 255, 255), 10)
                else:
                    distance_text = (f"MAX_DISTANCE = {max(max_distance)} pixel "
                                     f" MIN_DISTANCE = {min(min_distance)} pixel"
                                     f"Error detection!!!")

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
                            
                # print(max(max_distance), min(min_distance), mean_distance)
                return lines1, lines2

        else:
            distance_text = (f"Without distance value.")

            cv2.putText(line_image, distance_text, (20, line_image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 10)
            print("Only have one line detection no distance")
    else:
        print("no line detection")

            