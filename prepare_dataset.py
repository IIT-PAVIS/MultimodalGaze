import torch
from torch.utils.data import Dataset
import pandas as pd
import sys, ast, os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pickle as pkl
sys.path.append('../')
from variables import Variables, RootVariables
from build_dataset import BUILDING_DATASETS
from torchvision import transforms

class IMU_GAZE_FRAME_DATASET:
    def __init__(self, test_folder, reset_dataset=0):
        self.var = RootVariables()
        self.dataset = BUILDING_DATASETS(test_folder)
        self.frame_datasets = None
        self.imu_train_datasets, self.gaze_train_datasets = None, None
        self.imu_test_datasets, self.gaze_test_datasets = None, None
        if Path(self.var.root + 'datasets/' + test_folder[5:] + '/imuExtracted_training_data' + '.npy').is_file():
            print('Files exists')
            self.imu_train_datasets = np.load(self.var.root + 'datasets/' + test_folder[5:] + '/imuExtracted_training_data' + '.npy')
            self.gaze_train_datasets = np.load(self.var.root + 'datasets/' + test_folder[5:] + '/gazeExtracted_training_data' + '.npy')
            self.imu_test_datasets = np.load(self.var.root + 'datasets/' + test_folder[5:] + '/imuExtracted_testing_data' + '.npy')
            self.gaze_test_datasets = np.load(self.var.root + 'datasets/' + test_folder[5:] + '/gazeExtracted_testing_data' + '.npy')
        else:
            print('saved files does not exis')
            self.imu_train_datasets, self.imu_test_datasets = self.dataset.load_unified_imu_dataset()
            self.gaze_train_datasets, self.gaze_test_datasets = self.dataset.load_unified_gaze_dataset()
            np.save(self.var.root + 'datasets/' + test_folder[5:] + '/imuExtracted_training_data' + '.npy', self.imu_train_datasets)
            np.save(self.var.root + 'datasets/' + test_folder[5:] + '/gazeExtracted_training_data' + '.npy', self.gaze_train_datasets)
            np.save(self.var.root + 'datasets/' + test_folder[5:] + '/imuExtracted_testing_data' + '.npy', self.imu_test_datasets)
            np.save(self.var.root + 'datasets/' + test_folder[5:] + '/gazeExtracted_testing_data' + '.npy', self.gaze_test_datasets)

        self.dataset.load_unified_frame_dataset(reset_dataset)

        self.gaze_train_datasets = self.gaze_train_datasets.reshape(-1, 4, self.gaze_train_datasets.shape[-1])
        self.imu_train_datasets = self.imu_train_datasets.reshape(-1, 4, self.imu_train_datasets.shape[-1])
        #
        self.gaze_test_datasets = self.gaze_test_datasets.reshape(-1, 4, self.gaze_test_datasets.shape[-1])
        self.imu_test_datasets = self.imu_test_datasets.reshape(-1, 4, self.imu_test_datasets.shape[-1])

    def __len__(self):
        return int(len(self.gaze_train_datasets))      ## number of frames corresponding to

if __name__ =="__main__":
    var = RootVariables()
    device = torch.device("cpu")
    trim_size = 150
    frame_size = 256
    datasets = IMU_GAZE_FRAME_DATASET(var.root, frame_size, trim_size)
    train_imu_dataset = datasets.imu_train_datasets
    test_imu_dataset = datasets.imu_test_datasets

    train_gaze_dataset = datasets.gaze_train_datasets
    test_gaze_dataset = datasets.gaze_test_datasets
    print(train_imu_dataset[0], test_imu_dataset[0])
    # folders_num, gaze_start_index, gaze_end_index, trim_size = 0, 0, 0, 150
    # imu_start_index, imu_end_index = 0, 0
    # utls = Helpers()
    # sliced_imu_dataset, sliced_gaze_dataset = None, None
    # for index, subDir in enumerate(sorted(os.listdir(var.root))):
    #     if 'imu_' in subDir:
    #         folders_num += 1
    #         print(subDir)
    #         subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
    #         os.chdir(var.root + subDir)
    #         capture = cv2.VideoCapture('scenevideo.mp4')
    #         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #         gaze_end_index = gaze_start_index + frame_count - trim_size*2
    #         imu_end_index = imu_start_index + frame_count - trim_size
    #         sliced_imu_dataset = uni_imu_dataset[imu_start_index: imu_end_index]
    #         sliced_gaze_dataset = uni_gaze_dataset[gaze_start_index: gaze_end_index]
    #         dataset = IMU_DATASET(sliced_imu_dataset, sliced_gaze_dataset, device)
    #         print(len(dataset))
    #         i, g = dataset[1]
    #         print(g/1000.0)
    #         print(i.shape, g.shape)
    #         print(i[0], i[-1])
    #
    #         gaze_start_index = gaze_end_index
    #         imu_start_index = imu_end_index
    #     if 'imu_CoffeeVendingMachine_S2' in subDir :
    #         break
    #
    # print(sliced_imu_dataset[0].shape)
    # print('\n')
