import numpy as np
import cv2
import scipy
import math
import matplotlib.pyplot as plt
from utils import (loadImages, getImagesPoints, getWorldPoints, displayCorners, getAllH,  
                   getB, getA, getRotationAndTrans, extractParamFromA, lossFunc, retrieveA, 
                   reprojectPointsAndGetError)

def main():
    square_side = 12.5
    folder_name = "./data"
    save_folder = "./output"
    images = loadImages(folder_name)
    h, w = [6,9]
    all_image_corners = getImagesPoints(images, h, w)
    world_corners = getWorldPoints(square_side, h, w)

    # displayCorners(images, all_image_corners, h, w)

    print("Calculating H for %d images", len(images))
    all_H_init = getAllH(all_image_corners, square_side, h, w)
    print("Calculating B")
    B_init = getB(all_H_init)
    print("Estimated B = ", B_init)
    print("Calculating A")
    A_init = getA(B_init)
    print("Initialized A = ",A_init)
    print("Calculating rotation and translation")
    all_RT_init = getRotationAndTrans(A_init, all_H_init)
    print("Init Kc")
    kc_init = np.array([0,0]).reshape(2,1)
    print("Initialized kc = ", kc_init)

    print("Optimizing ...")
    x0 = extractParamFromA(A_init, kc_init)

    res = scipy.optimize.least_squares(fun=lossFunc, x0=x0, method="lm", args=[all_RT_init, all_image_corners, world_corners])
    x1 = res.x
    AK = retrieveA(x1)
    A_new = AK[0]
    kc_new = AK[1]

    previous_error, _ = reprojectPointsAndGetError(A_init, kc_init, all_RT_init, all_image_corners, world_corners)
    att_RT_new = getRotationAndTrans(A_new, all_H_init)
    new_error, points = reprojectPointsAndGetError(A_new, kc_new, att_RT_new, all_image_corners, world_corners)

    print("The error befor optimization was ", previous_error)
    print("The error after optimization is ", new_error)
    print("The A matrix is: ", A_new)

    K = np.array(A_new, np.float32).reshape(3,3)
    D = np.array([kc_new[0].squeeze(0),kc_new[1].squeeze(0), 0, 0] , np.float32)
    for i,image_points in enumerate(points):
        image = cv2.undistort(images[i], K, D)
        for point in image_points:
            x = int(point[0])
            y = int(point[1])
            image = cv2.circle(image, (x, y), 5, (0, 0, 255), 3)
        cv2.imshow('frame', image)
        filename = save_folder + str(i) + "reproj.png"
        cv2.imwrite(filename, image)
        cv2.waitKey()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()