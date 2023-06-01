import numpy as np
import cv2
import matplotlib.pyplot as plt
from Marker_Detection import get_MarkerPoints
from get_points import get_point

time_stamp_path_1 = './seq1/dataset/1681710717_541398178'
time_stamp_path_2 = './seq1/dataset/1681710717_577611807'
cam_direction_1 = np.loadtxt(f'{time_stamp_path_1}/camera.csv', dtype=str, delimiter=',')
cam_direction_1 = str(cam_direction_1).split('/')[2].split('_')[2]

sift = cv2.SIFT_create()

marker_img_1 = cv2.imread(f'{time_stamp_path_1}/raw_image.jpg', cv2.IMREAD_GRAYSCALE)
marker_pts_1 = get_MarkerPoints(time_stamp_path_1)
_, des1 = sift.compute(marker_img_1, marker_pts_1)

marker_img_2 = cv2.imread(f'{time_stamp_path_2}/raw_image.jpg', cv2.IMREAD_GRAYSCALE)
marker_pts_2 = get_MarkerPoints(time_stamp_path_2)
_, des2 = sift.compute(marker_img_2, marker_pts_2)

bf_matcher = cv2.BFMatcher()
matches = bf_matcher.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(marker_img_1, marker_pts_1, marker_img_2, marker_pts_2, good, None, flags=2)

plt.imshow(img3)
plt.show()