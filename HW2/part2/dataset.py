import os
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloader(dataset_dir, batch_size=1, split='test'):
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            ##### TODO: Data Augmentation Begin #####
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.GaussianBlur(3,3),
            ##### TODO: Data Augmentation End #####
            transforms.ToTensor(),
            transforms.Normalize(means,stds)
        ])
    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

    dataset = CIFAR10Dataset(dataset_dir, split=split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=0, pin_memory=True, drop_last=(split=='train'))

    return dataloader

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None):
        super(CIFAR10Dataset).__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)
        
        self.image_names = json_data['filenames']
        if self.split != 'test':
            self.labels = json_data['labels']

        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):

        ########################################################
        # TODO:                                                #
        # Define the CIFAR10Dataset class:                     #
        #   1. use Image.open() to load image according to the # 
        #      self.image_names                                #
        #   2. apply transform on image                        #
        #   3. if not test set, return image and label with    #
        #      type "long tensor"                              #
        #   4. else return image only                          #
        #                                                      #
        # NOTE:                                                #
        # You will not have labels if it's test set            #
        ########################################################

        img = Image.open(os.path.join(self.dataset_dir,self.image_names[index]))
        img_transform = self.transform(img)
        if self.split != 'test':
            return {'images': img_transform, 'labels': self.labels[index]}
        else:
            return {'images': img_transform} 
        # return {
        #     'images': image, 
        #     'labels': label
        # }
