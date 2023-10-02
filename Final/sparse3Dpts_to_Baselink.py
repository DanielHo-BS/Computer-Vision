import numpy as np
import os
from get_Matrix import convert_toMatrix, get_ExtrinsicMatrix

def get_direction(file_path):
    cam_direction = np.loadtxt(f'{file_path}/camera.csv', dtype=str, delimiter=',')
    cam_direction = str(cam_direction).split('/')[2].split('_')[2]

    return cam_direction

def baselink_to_World(world_camera, camera_baselink):
    if np.array(world_camera).shape[0] != 4:
        world_camera = np.concatenate((np.array(world_camera), np.array([[0,0,0,1]])), axis=0)
    if np.array(camera_baselink).shape[0] != 4:
        camera_baselink = np.concatenate((np.array(camera_baselink), np.array([[0,0,0,1]])), axis=0)
    
    world_baselink = np.dot(world_camera, camera_baselink)
    baselink_world = np.linalg.inv(world_baselink)
    delta_x = baselink_world[0, 3]
    delta_y = baselink_world[1, 3]

    return baselink_world, delta_x, delta_y

sequence = 'seq2'  # change different sequence
file_path = f'./output_{sequence}_alltime'
# ID_to_Name = {}
# Name_with_TxTy = {}
Name_with_TxTyTzQ = {}

directions = ['f', 'fl', 'fr', 'b']
for dir in directions:
    with open(f'{file_path}/output_{sequence}_{dir}/images.txt', 'r') as lines:
        for line in lines:
            if '#' in line:
                continue
            data = line.split(' ')
            if len(data) == 10:
                # ID_to_Name[data[0]] = data[9].split('\n')[0][:-4]  # data[0]: ID, data[9]: Name
                # Name_with_TxTy[data[9].split('\n')[0][:-4]] = [data[5], data[6]]  # Tx, Ty
                Name_with_TxTyTzQ[data[9].split('\n')[0][:-4]] = [float(data[5]), float(data[6]), float(data[7]),
                                                                  float(data[2]), float(data[3]), float(data[4]), float(data[1])]  # tx, ty, tz, qx, qy, qz, qw

cam_info_path = '../ITRI_dataset/camera_info/lucid_cameras_x00'
ExtMatrix = get_ExtrinsicMatrix(cam_info_path)

Final_TxTyTzQ = {}
with open(f'../ITRI_dataset/{sequence}/localization_timestamp.txt', 'r') as lines:
    for idx, line in enumerate(lines):
        local_ts = line.split('\n')[0]
        if local_ts in Name_with_TxTyTzQ.keys():
            # Transform matrix: [R|T], shape: 3x4
            Final_TxTyTzQ[local_ts] = convert_toMatrix(Name_with_TxTyTzQ[local_ts])
            if get_direction(f'../ITRI_dataset/{sequence}/dataset/{local_ts}') == 'f':
                TransMatrix = ExtMatrix['base_link-f']

            elif get_direction(f'../ITRI_dataset/{sequence}/dataset/{local_ts}') == 'fl':
                b_f = np.concatenate((ExtMatrix['base_link-f'], np.array([[0,0,0,1]])), axis=0)
                f_fl = np.concatenate((ExtMatrix['f-fl'], np.array([[0,0,0,1]])), axis=0)
                TransMatrix = np.dot(b_f, f_fl)

            elif get_direction(f'../ITRI_dataset/{sequence}/dataset/{local_ts}') == 'fr':
                b_f = np.concatenate((ExtMatrix['base_link-f'], np.array([[0,0,0,1]])), axis=0)
                f_fr = np.concatenate((ExtMatrix['f-fr'], np.array([[0,0,0,1]])), axis=0)
                TransMatrix = np.dot(b_f, f_fr)

            elif get_direction(f'../ITRI_dataset/{sequence}/dataset/{local_ts}') == 'b':
                b_f = np.concatenate((ExtMatrix['base_link-f'], np.array([[0,0,0,1]])), axis=0)
                f_fl = np.concatenate((ExtMatrix['f-fl'], np.array([[0,0,0,1]])), axis=0)
                fl_b = np.concatenate((ExtMatrix['fl-b'], np.array([[0,0,0,1]])), axis=0)
                TransMatrix = np.dot(b_f, np.dot(f_fl, fl_b))
            
            _, dx, dy = baselink_to_World(Final_TxTyTzQ[local_ts], TransMatrix)
            Final_TxTyTzQ[local_ts] = [dx, dy]

        else:
            Final_TxTyTzQ[local_ts] = [0, 0]

if not os.path.exists(f'./solution/{sequence}/'):
    os.makedirs(f'./solution/{sequence}/')
count = 0
for key, val in Final_TxTyTzQ.items():
    # print(key, val)
    with open(f'./solution/{sequence}/pred_pose.txt', 'a') as file:
        if count == len(Final_TxTyTzQ.keys()) - 1:
            file.write(f'{val[0]} {val[1]}')
        else:
            file.write(f'{val[0]} {val[1]}\n')
        count += 1