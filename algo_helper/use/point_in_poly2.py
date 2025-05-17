"""
by zdy
Determine if a point is inside a given polygon or not.

The algorithm is called the "Ray Casting Method".
Source: http://geospatialpython.com/2011/01/point-in-polygon.html
"""
import time
import numpy as np
import cv2

# roi = [[679, 209], [930, 185], [1280, 586], [1270, 681], [676, 692]]
# point_in_poly_single(749, 718, roi)
def point_in_poly_single(x, y, poly):
    """                                                                                                                                                                  
    Determine if a point is inside a given polygon or not.                                                                                                               
                                                                                                                                                                         
    Polygon is a list of (x, y) pairs.                                                                                                                                    
    This function returns True or False.                                                                                                                                 
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def point_in_poly(x, y, polylist):
    """
    Determine if a point is inside a given polygon or not.

    Polygon is a list of (x, y) pairs.
    This function returns True or False.
    """
    result = False
    for poly in polylist:
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        if (inside is True):
            return True

    return result


class PointInROIChecker:
    def __init__(self, roi, max_h, max_w):
        """
        roi = [[679, 209], [930, 185], [1280, 586], [1270, 681], [676, 692]] # (xmax=1280, ymax=720)
        shape = [720, 1280] # (height, width)
        """
        shape = [max_h, max_w]
        mask = np.zeros(shape, dtype=np.uint8)
        roi_np = np.array([roi], dtype=np.int32)[:, ::-1] # (x, y) -> (y, x) for opencv.
        cv2.fillPoly(mask, roi_np, 1) # 255 for visualization.
        
        self.mask_list_python = mask.tolist()

    def point_in_poly_single(self, point_x, point_y):
        flag_in_roi = bool(self.mask_list_python[point_y][point_x])
        return flag_in_roi



if __name__ == "__main__":    
    ## 1. Init ROI
    roi = [[679, 209], [930, 185], [1280, 586], [1270, 681], [676, 692]] # (xmax=1280, ymax=720)
    shape = [720, 1280] # (height, width)
    ## 2. Init Testing Point
    point = (749, 718) # (x, y).
    point_x = point[0]
    point_y = point[1]

    ## 3. Init Class 
    ROIChecker_1 = PointInROIChecker(roi=roi, max_h=shape[0], max_w=shape[1])

    # Compare the performances.

    a = time.time()
    for _ in range(10000):
        in_roi = point_in_poly_single(point_x, point_y, roi)
    b = time.time()
    for _ in range(10000):
        in_roi_opencv = ROIChecker_1.point_in_poly_single(point_x, point_y)
    c = time.time()

    
    d = time.time()
    print(in_roi, in_roi_opencv, b-a, c-b) # False 1.0967254638671875e-05
    print((b-a)/(c-b))
    

    # UNITEST
    # mask_vis_unittest = np.zeros(shape+[3], dtype=np.uint8)
    # for xx in range(1280):
    #     for yy in range(720):
    #         point = (xx, yy) # (x, y).
    #         point_x = point[0]
    #         point_y = point[1]
    #         in_roi = point_in_poly_single(point_x, point_y, roi)
    #         in_roi_opencv = ROIChecker_1.point_in_poly_single(point_x, point_y)
    #         # assert (in_roi == in_roi_opencv),  f"{in_roi} != {in_roi_opencv} at ({point_x}, {point_y})"
    #         if (in_roi != in_roi_opencv):
    #             print(f"{in_roi} != {in_roi_opencv} at ({point_x}, {point_y})")
    #             cv2.circle(mask_vis_unittest, (xx, yy), 1, (0, 0, 255), -1)
    # cv2.imwrite("mask_diff.jpg", mask_vis_unittest)






    # Unitest.
    # mask_vis_unittest = np.zeros(shape+[3], dtype=np.uint8)
    # for xx in range(1280):
    #     for yy in range(720):
    #         point = (xx, yy) # (x, y).
    #         point_x = point[0]
    #         point_y = point[1]
    #         in_roi = point_in_poly_single(point_x, point_y, roi)
    #         in_roi_opencv = mask[point_y, point_x]

    #         if in_roi == True and in_roi_opencv == False:
    #             cv2.circle(mask_vis_unittest, (xx, yy), 1, (0, 0, 255), -1)
    #         elif in_roi == False and in_roi_opencv == True:
    #             cv2.circle(mask_vis_unittest, (xx, yy), 1, (0, 255, 0), -1)
    
    # cv2.imwrite("mask_diff.jpg", mask_vis_unittest)
    # cv2.imshow("diff", mask_vis_unittest)
    # cv2.waitKey()
