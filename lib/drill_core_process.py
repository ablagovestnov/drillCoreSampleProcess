import math
import random
from functools import reduce

import cv2
import numpy as np
import pandas as pd


class DrillCoreProcess():
    hole = []

    def __init__(self, images_path='images/'):
        self.images_path = images_path
        self.fill_drill_hole()
    def process_image(self, image_name='', metrics={'start':0, 'end': 0}):
        location = self.images_path + image_name
        original = cv2.imread(location)
        image = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
        cores = []

        height = image.shape[0]
        width = image.shape[1]

        kernel = np.ones((6, 6), 'uint8')
        image = cv2.erode(image, kernel, iterations=1)

        src = image
        dst = cv2.Canny(src, 50, 200, None, 3)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        if lines is not None:
            resulted_lines = []
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                if math.dist(pt1, pt2) > 600 and (-0.001 < a < 0.001):
                    resulted_lines.append(round((pt1[1] + pt2[1]) / 2))
                    # cv2.line(
                    #     original,
                    #     (0, pt1[1]),
                    #     (width, pt2[1]),
                    #     (0, 0, 255), 3, cv2.LINE_AA)
            resulted_lines.sort()
            prev_y = 0
            clasters = []
            cur_claster = []
            for line_y in resulted_lines:
                if (line_y - prev_y) < 20  :
                    cur_claster.append(line_y)
                else:
                    if len(cur_claster):
                        clasters.append(cur_claster)
                        cur_claster = []
                prev_y = line_y
            clasters.append(cur_claster)
            clasters = list(
                map(lambda c: (max(c), min(c), round(sum(c)/len(c))), clasters)
            )

            for i in range(0, len(clasters)-1):
                if i > 0 and (clasters[i][1] - clasters[i-1][0]) < 50:
                    continue
                piece = original[clasters[i][0]:clasters[i+1][1]:]
                cores.append(piece)
                cv2.line(
                    original,
                    (0, clasters[i][2]),
                    (width, clasters[i][2]),
                    (0, 255, 255), 3, cv2.LINE_AA)
            cv2.line(
                original,
                (0, clasters[len(clasters)-1][2]),
                (width, clasters[len(clasters)-1][2]),
                (0, 255, 255), 3, cv2.LINE_AA)
            # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", original)
            cv2.imwrite('lines_'+image_name, original)
            # cv2.waitKey()
            max_piece_h = max(list(map(lambda core: core.shape[0] ,cores)))
            concat_img = cv2.rotate(
                cv2.hconcat(
                    list(
                        map(lambda core_img: cv2.resize(core_img[:, 28:-28], (core_img.shape[1], max_piece_h)), cores)
                    )
                ),
                cv2.ROTATE_90_CLOCKWISE,
            )
            min_l = 190.7
            max_l = 193.45
            core_sample_len = max_l - min_l
            core_sample_len_portion = round(concat_img.shape[0]/(max_l - min_l))
            print(core_sample_len_portion)

            start_metric = self.find_closest_meter(min_l)
            end_metric = self.find_closest_meter(max_l)
            # print(start_metric[1])
            text_color = (255, 255, 255)
            for i in range(start_metric[1], end_metric[1]+1):
                int_s = round(core_sample_len_portion*(self.hole[i]['interval'] - min_l))
                cv2.rectangle(concat_img, (0, int_s+75+2), (concat_img.shape[1], int_s + 142+75), (0, 0, 0), -1)
                cv2.line(
                    concat_img,
                    (0, int_s if int_s <= concat_img.shape[0] else concat_img.shape[0]),
                    (concat_img.shape[1], int_s if int_s <= concat_img.shape[0] else concat_img.shape[0]),
                    (0, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    concat_img,
                    self.hole[i]['mineralization'],
                    (5, int_s+75+35 if int_s+35 <= concat_img.shape[0] else concat_img.shape[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=text_color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    concat_img,
                    str(self.hole[i]['interval']),
                    (5, int_s+75+70 if int_s+70 <= concat_img.shape[0] else concat_img.shape[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=text_color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    concat_img,
                    '-',
                    (5, int_s+75+105 if int_s+105 <= concat_img.shape[0] else concat_img.shape[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=text_color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    concat_img,
                    str(self.hole[i+1]['interval']),
                    (5, int_s+75+140 if int_s+140 <= concat_img.shape[0] else concat_img.shape[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=text_color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                # +' ('+str(self.hole[i]['interval'])+' - '+str(self.hole[i+1]['interval'])+')'
                # print(self.hole[i+1]['interval'] - )
            #
            cv2.imshow('Concat image', concat_img)
            cv2.imwrite('concat_'+image_name, concat_img)
            cv2.waitKey()
        # cv2.waitKey()
        cv2.destroyAllWindows()

    def find_closest_meter(self, meter):
        closest = None
        closest_index = None
        # print('To find', float(meter))
        for i in range(0, len(self.hole)):
            portion = self.hole[i]
            if (not closest) or (abs(float(portion['interval']) - float(meter)) <= abs(float(closest) - float(meter))):
                closest = portion['interval']
                closest_index = i
            if float(portion['interval']) - float(meter) > 2:
                break
        return (closest, closest_index)

    def fill_drill_hole(self):
        table = pd.read_excel('U-DD-113_Log.xlsm', index_col=None, na_values=['NA'], usecols = "B,F,G", skiprows=5)
        table = table.to_dict()
        min = 190.7
        max = 193.45
        ln = len(table[1])
        # print(ln)
        hole = []
        for i in range(0, ln):
            hole.append({
                'interval': table[1][i],
                'mineralization': table[5][i],
                'color': table[6][i],
            })
        hole.sort(key=lambda i: i['interval'])
        self.hole = hole
        # print(hole)
        # for row in table.keys():
        #     print(row, len(table[row].keys()))
        # print(table)
        # print(self.find_closest_meter(min))
        # print(self.find_closest_meter(max))

        # print(list(hole[190:200]))