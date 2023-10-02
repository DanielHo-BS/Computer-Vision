import numpy as np
import cv2
from get_points import ImgPts
from tqdm import tqdm
import os


def mkdir(dir_path):
    if os.path.exists(dir_path) != True:
        os.makedirs(dir_path)

sequence = 'test1'  # change sequence
txt_path = f"./ITRI_dataset/{sequence}/all_timestamp.txt"
time_stamp_dict = {}
with open(txt_path, 'r') as f:
    lines = f.readlines()

save_path = f'./ITRI_dataset/{sequence}_all_reconstruction/kpt_des'
direction = ['f', 'fl', 'fr', 'b']
# if os.path.exists(f'{save_path}/{ts}_kpt_des.txt'):
#     os.remove(f'{save_path}/{ts}_kpt_des.txt')
for d in direction:
    mkdir(f'{save_path}/{d}')

for line in tqdm(lines):
    ts = line.strip("\n")

    ## update
    ############ read which camera is used ######################
    with open (os.path.join(f"./ITRI_dataset/{sequence}"+"/"+"dataset/"+ ts +"/camera.csv"),'r') as can :
        can_line = can.readlines()
    can_angle = can_line[0][-5]

    if can_angle == "f" :
        img_mask = "gige_100_f_hdr_mask.png"
    elif can_angle == "b" :
        img_mask = "gige_100_b_hdr_mask.png"
    elif can_angle == "l" :
        img_mask = "gige_100_fl_hdr_mask.png"
    elif can_angle == "r" :
        img_mask = "gige_100_fr_hdr_mask.png"

    car_mask = os.path.join("./ITRI_dataset/camera_info/lucid_cameras_x00",img_mask)
    ##############################################################


    dataset_path = f'./ITRI_dataset/{sequence}/dataset/{ts}'
    # if os.path.exists(f'{save_path}/{ts}_kpt_des.txt'):
    #     os.remove(f'{save_path}/{ts}_kpt_des.txt')

    img_2Dpts = ImgPts(dataset_path, ts, car_mask)
    marker_pts = []
    for i in range(len(img_2Dpts[0])):
        img_2Dpt = np.array([img_2Dpts[0][i], img_2Dpts[1][i], 1]).T  # [x, y, 1]
        marker_pts.append(cv2.KeyPoint(int(img_2Dpts[0][i]), int(img_2Dpts[1][i]), 1))
    # print(marker_pts)

    sift = cv2.SIFT_create()
    marker_img = cv2.imread(f'./ITRI_dataset/{sequence}/dataset/{ts}/raw_image.jpg', cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.compute(marker_img, marker_pts)

    if can_angle == 'r':
        can_angle = 'fr'
    if can_angle == 'l':
        can_angle = 'fl'

    with open(f'{save_path}/{can_angle}/{ts}.jpg.txt', 'a') as file:
        file.write(str(len(keypoints)) + ' ' + '128' + '\n')
    
    for idx, kpt in enumerate(keypoints):
        sift_pt_des = []
        # print(kpt.pt)    # (x, y)
        # print(kpt.size)  # scale
        # print(kpt.angle) # orientation
        sift_pt_des.extend([kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle])
        sift_pt_des.extend(descriptors[idx])

        with open(f'{save_path}/{can_angle}/{ts}.jpg.txt', 'a') as file:
            line = ' '.join(str(data) for data in sift_pt_des)
            file.write(line + '\n')
    