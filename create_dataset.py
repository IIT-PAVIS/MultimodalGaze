import sys, os
import numpy as np
import torch.nn as nn
import cv2
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
sys.path.append('../')
# from FlowNetPytorch.models import FlowNetS
from variables import RootVariables
from helpers import standarization

class All_Dataset:
    def __init__(self):
        self.var = RootVariables()

    def get_dataset(self, original_img_csv, feat, labels, index):
        if index == 0:
            return self.SIG_FINAL_DATASET(feat, labels)
        elif index == 1:
            return self.VIS_FINAL_DATASET(original_img_csv, labels)
        else:
            return self.FUSION_DATASET(original_img_csv, feat, labels)

    class FUSION_DATASET(Dataset):
        def __init__(self, original_img_csv, imu_feat, labels):
            self.imu_data, self.gaze_data = [], []
            self.indexes = []
            self.var = RootVariables()
            self.ori_imgs_path = pd.read_csv(self.var.root + original_img_csv + '.csv')
            name_index = 5 if len(self.ori_imgs_path.iloc[0, 1].split('/')) > 5 else 4
            subfolder = self.ori_imgs_path.iloc[0, 1].split('/')[name_index]
            f_index = 7
            checkedLast = False
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                imu_check = np.isnan(imu_feat[index])
                if check.any() or imu_check.any():
                    f_index += 1
                else:
                    f_index += 1
                    self.gaze_data.append(labels[index])
                    if self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index] == subfolder:
                        self.indexes.append(f_index)
                    else:
                        f_index += 8#1
                        self.indexes.append(f_index)
                        subfolder = self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index]
                    self.imu_data.append(imu_feat[index])

            self.imu_data = standarization(self.imu_data)

            assert len(self.imu_data) == len(self.indexes)
            assert len(self.gaze_data) == len(self.indexes)

            self.transforms = transforms.Compose([transforms.ToTensor()])

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
            ##Imgs
            for i in range(f_index-8, f_index +1, 1):
                # print(self.ori_imgs_path.iloc[i, 1], f_index)
                img = torch.cat((img, self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=1)), 1) if i > f_index-8 else self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=1)

            targets = self.gaze_data[index]
            targets[:,0] *= 512.0
            targets[:,1] *= 288.0

            return (img).to("cuda:0"), torch.from_numpy(self.imu_data[index]).to("cuda:0"), torch.from_numpy(targets).to("cuda:0")

    class VIS_FINAL_DATASET(Dataset):
        def __init__(self, original_img_csv, labels):
            self.gaze_data = []
            self.indexes = []
            self.var = RootVariables()
            self.ori_imgs_path = pd.read_csv(self.var.root + original_img_csv + '.csv')
            name_index = 5 if len(self.ori_imgs_path.iloc[0, 1].split('/')) > 5 else 4
            subfolder = self.ori_imgs_path.iloc[0, 1].split('/')[name_index]
            f_index = 7
            checkedLast = False
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                if check.any():
                    f_index += 1
                    # continue
                else:
                    f_index += 1
                    self.gaze_data.append(labels[index])
                    if self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index] == subfolder:
                        self.indexes.append(f_index)
                    else:
                        f_index += 8#1
                        self.indexes.append(f_index)
                        subfolder = self.ori_imgs_path.iloc[f_index, 1].split('/')[name_index]

            self.transforms = transforms.ToTensor()
            assert len(self.gaze_data) == len(self.indexes)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.indexes) # len(self.labels)

        def __getitem__(self, index):
            f_index = self.indexes[index]
            ##Imgs
            for i in range(f_index-8, f_index +1, 1):
                # print(self.ori_imgs_path.iloc[i, 1], f_index)
                img = torch.cat((img, self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=1)), 1) if i > f_index-8 else self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=1)

            # for i in range(f_index, f_index-5, -1):
            #     print(self.ori_imgs_path.iloc[i, 1], f_index)
            #     img = torch.cat((img, self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=1)), 1) if i < f_index else self.transforms(Image.open(self.ori_imgs_path.iloc[i, 1])).unsqueeze(dim=1)
            # print(img.shape)
            targets = self.gaze_data[index]
            #targets[:,0] *= 0.2667
            #targets[:,1] *= 0.3556

            targets[:,0] *= 512.0
            targets[:,1] *= 288.0

            return (img).to("cuda:0"), torch.from_numpy(targets).to("cuda:0")

    class SIG_FINAL_DATASET(Dataset):
        def __init__(self, feat, labels):
            self.gaze_data, self.imu_data = [], []
            checkedLast = False
            for index in range(len(labels)):
                check = np.isnan(labels[index])
                imu_check = np.isnan(feat[index])
                if check.any() or imu_check.any():
                    continue
                else:
                    self.gaze_data.append(labels[index])
                    self.imu_data.append(feat[index])

            self.imu_data = standarization(self.imu_data)

            assert len(self.imu_data) == len(self.gaze_data)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def __len__(self):
            return len(self.gaze_data) # len(self.labels)

        def __getitem__(self, index):
            targets = self.gaze_data[index]
            targets[:,0] *= 512.0
            targets[:,1] *= 288 #384 #288.0 # 384

            return torch.from_numpy(self.imu_data[index]).to(self.device), torch.from_numpy(targets).to(self.device)
