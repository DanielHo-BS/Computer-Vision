import shutil
import numpy as np
import os

sequence = 'test1'  # change different sequence
txt_path = f"./ITRI_dataset/{sequence}/all_timestamp.txt"
try:
    with open(txt_path, 'r') as f:
        lines = f.readlines()
except IOError:
    print('Error: open txt')
save_path = f'./ITRI_dataset/{sequence}_all_reconstruction'

direction = ['f', 'fl', 'fr', 'b']
for d in direction:
    if os.path.exists(f'{save_path}/{d}_img') != True:
        os.makedirs(f'{save_path}/{d}_img')

for line in lines:
    tsp = line.strip("\n")  # time stamp path
    cam_direction = np.loadtxt(f'./ITRI_dataset/{sequence}/dataset/{tsp}/camera.csv', dtype=str, delimiter=',')
    cam_direction = str(cam_direction).split('/')[2].split('_')[2]
    if cam_direction == 'f':
        shutil.copyfile(f'./ITRI_dataset/{sequence}/dataset/{tsp}/raw_image.jpg', f'{save_path}/f_img/{tsp}.jpg')
        with open(f'{save_path}/f_image.txt', 'a') as file:
            file.write(f'{tsp}.jpg' + '\n')
    if cam_direction == 'fl':
        shutil.copyfile(f'./ITRI_dataset/{sequence}/dataset/{tsp}/raw_image.jpg', f'{save_path}/fl_img/{tsp}.jpg')
        with open(f'{save_path}/fl_image.txt', 'a') as file:
            file.write(f'{tsp}.jpg' + '\n')
    if cam_direction == 'fr':
        shutil.copyfile(f'./ITRI_dataset/{sequence}/dataset/{tsp}/raw_image.jpg', f'{save_path}/fr_img/{tsp}.jpg')
        with open(f'{save_path}/fr_image.txt', 'a') as file:
            file.write(f'{tsp}.jpg' + '\n')
    if cam_direction == 'b':
        shutil.copyfile(f'./ITRI_dataset/{sequence}/dataset/{tsp}/raw_image.jpg', f'{save_path}/b_img/{tsp}.jpg')
        with open(f'{save_path}/b_image.txt', 'a') as file:
            file.write(f'{tsp}.jpg' + '\n')
