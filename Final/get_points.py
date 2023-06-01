import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def nms(bboxes, conf, iou_thresh):
    bboxes, conf = np.array(bboxes), np.array(conf)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    result = []
    index = conf.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        result.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]

    return bboxes[result].tolist(), conf[result].tolist()


def get_bboxes(csv_path, nms_threshold=0.2):
    bboxes = []
    conf = []
    try:
        with open(csv_path, "r") as csvfile:
            pred_yolo = csv.reader(csvfile)
            for pred in pred_yolo:  # bboxes(x11y11x22y22), confidence
                pred = list(map(float, pred))
                if pred[5] <= 0.2:  # skip, when confidence is lower than 0.2.
                    continue
                else:
                    bboxes.append(pred[:4])
                    conf.append(pred[5])

        bboxes, conf = nms(bboxes, conf, nms_threshold)
    except IOError:
        print("Error: open csv")
    except IndexError:  # skip, when  result of detect is nothing.
        pass
    return bboxes


def crop(img, bboxes):
    # init
    img_crops, means, lower = [], [], []
    upper = np.array([255, 255, 255])

    for bbox in bboxes:
        for i in range(4):
            bbox[i] = np.clip(bbox[i], 1, float("inf"))
        signs = img[
            int(bbox[1]) - 1 : int(bbox[3]) + 1, (int(bbox[0])) - 1 : int(bbox[2]) + 1
        ]
        std = int(np.std(signs.flatten()))
        mean = int(signs.flatten().mean()) + std
        img_crops.append(signs)
        means.append(mean)
        if mean < 250:
            lower.append(np.array([mean, mean, mean]))
        else:
            lower.append(np.array([175, 175, 175]))

    return {"img_crops": img_crops, "means": means, "lower": lower, "upper": upper}


def get_point(img, crops, file):
    # init
    img_crops = crops["img_crops"]
    means = crops["means"]
    lower = crops["lower"]
    upper = crops["upper"]
    special_points = [[], []]  # special_points[0]=x, special_points[1]=y
    otsu_threshold, _ = cv2.threshold(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # create save dir
    save_dir = "outputs/" + file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for id, img_crop in enumerate(img_crops):
        mask = cv2.inRange(img_crop, lower[id], upper)
        img_mask = cv2.bitwise_and(img_crop, img_crop, mask=mask)
        gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        # find the contours with binary image
        _, img_bn = cv2.threshold(
            gray, means[id], 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, contours, _ = cv2.findContours(
            img_bn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # find the canny of each crop, for noise filtering
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        canny = cv2.Canny(gaussian, threshold1=otsu_threshold, threshold2=175)

        for contour in contours:
            # find corner points of each contour
            corners = cv2.approxPolyDP(contour, 3, True)
            for corner in corners:
                if canny[corner[0][1]][corner[0][0]] != 0:
                    cv2.circle(
                        img_crop, (corner[0][0], corner[0][1]), 1, [0, 0, 255], 3
                    )
                    special_points[0].append(corner[0][0])
                    special_points[1].append(corner[0][1])
                else:
                    continue  # skip, when corner not in the canny

        cv2.imwrite(save_dir + "/canny" + str(id) + ".png", canny)  # save the canny of each crop
    cv2.imwrite(save_dir + "/points.png", img)  # save the road marker
    return special_points


def main(dataset_path, ts):
    csv_path = dataset_path + "/detect_road_marker.csv"
    img_path = dataset_path + "/raw_image.jpg"
    img = cv2.imread(img_path)
    bboxes = get_bboxes(csv_path)
    crops = crop(img, bboxes)
    sp = get_point(img, crops, ts)
    return sp


if __name__ == "__main__":
    sequence = "./ITRI_dataset/seq1"
    ts_path = sequence + "/all_timestamp.txt"
    try:
        with open(ts_path, "r") as f:
            lines = f.readlines()
    except IOError:
        print("Error: open txt")
    ts = lines[1113].strip("\n")
    dataset_path = f"{sequence}/dataset/{ts}"
    sp = main(dataset_path, ts)
    print(sp[0], '\n', sp[1])
