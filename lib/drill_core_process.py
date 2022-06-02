import math
import random
from functools import reduce

import cv2
import numpy as np


class DrillCoreProcess():
    def __init__(self, imagesPath='images/', imageName=''):
        original = cv2.imread(imagesPath + imageName)
        self.image = cv2.imread(imagesPath + imageName, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((6, 6), 'uint8')
        self.image = cv2.erode(self.image, kernel, iterations=1)
        height = self.image.shape[0]
        width = self.image.shape[1]
        src = self.image

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
            resulted_lines.sort()
            prev_y = 0
            clasters = []
            cur_claster = []
            for line_y in resulted_lines:
                if line_y - prev_y < 10:
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
                piece = original[clasters[i][0]:clasters[i+1][1]:]
                cv2.imshow('Piece', piece)
                cv2.imwrite('imgPiece'+str(i)+'.jpeg', piece)
                cv2.line(
                    original,
                    (0, clasters[i][2]),
                    (width, clasters[i][2]),
                    (0, 0, 255), 3, cv2.LINE_AA)
                cv2.waitKey()
                # img_pieces.append()
            cv2.line(
                original,
                (0, clasters[len(clasters)-1][2]),
                (width, clasters[len(clasters)-1][2]),
                (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", original)
            cv2.imwrite('imgLines.jpeg', original)

        cv2.waitKey()

        cv2.destroyAllWindows()
