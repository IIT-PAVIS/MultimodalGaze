import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import argparse
import random
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
sys.path.append('../')
from prepare_dataset import IMU_GAZE_FRAME_DATASET
from variables import RootVariables
from signal_pipeline import IMU_PIPELINE, IMU_DATASET
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":
    pipeline = IMU_PIPELINE()

    for index, subDir in enumerate(sorted(os.listdir(pipeline.var.root))):
        if 'train_' in subDir:
            print(subDir)
            pipeline.start_index, pipeline.end_index = 0, 0
            subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
            os.chdir(pipeline.var.root + subDir)
            capture = cv2.VideoCapture('scenevideo.mp4')
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            pipeline.gaze_end_index = pipeline.gaze_start_index + frame_count - pipeline.var.trim_frame_size*2
            pipeline.imu_end_index = pipeline.imu_start_index + frame_count - pipeline.var.trim_frame_size

            sliced_imu_dataset = pipeline.train_imu_dataset[pipeline.imu_start_index: pipeline.imu_end_index]
            sliced_gaze_dataset = pipeline.train_gaze_dataset[pipeline.gaze_start_index: pipeline.gaze_end_index]

            fig = plt.figure()
            sliced_gaze_dataset = sliced_gaze_dataset.reshape(-1, 2)
            sliced_imu_dataset = sliced_imu_dataset.reshape(-1, 6)
            order = 1
            fs = 100.0       # sample rate, Hz
            cutoff = 1.667
            x = np.arange(len(sliced_imu_dataset))
            # fig.add_subplot(221)
            plt.plot(x, sliced_imu_dataset[:,4])

            # sliced_imu_dataset[:,0] = pipeline.butter_lowpass_filter(sliced_imu_dataset[:,0], cutoff, fs, order)
            # sliced_imu_dataset[:,1] = pipeline.butter_lowpass_filter(sliced_imu_dataset[:,1], cutoff, fs, order)
            # sliced_imu_dataset[:,2] = pipeline.butter_lowpass_filter(sliced_imu_dataset[:,2], cutoff, fs, order)
            # sliced_imu_dataset[:,3] = pipeline.butter_lowpass_filter(sliced_imu_dataset[:,3], cutoff, fs, order)
            sliced_imu_dataset[:,4] = pipeline.butter_lowpass_filter(sliced_imu_dataset[:,4], cutoff, fs, order)
            # sliced_imu_dataset[:,5] = pipeline.butter_lowpass_filter(sliced_imu_dataset[:,5], cutoff, fs, order)

            # y = butter_lowpass_filter(sliced_imu_dataset[:,1], cutoff, fs, order)
            # print(len(y), len(sliced_imu_dataset[:,0]))
            # fig.add_subplot(222)
            plt.plot(x, sliced_imu_dataset[:,4])
            # plt.plot(x[50:], y[50:])

            fig.set_size_inches(25, 15)
            # plt.savefig('/home/sans/Downloads/gaze_data/data_plots/' + subDir[:-1] + '_lowpass_filter.png')
            plt.show()


            # if 'tescsdvt_' in subDir:
            #     nan_index = []
            #     sliced_gaze_dataset = sliced_gaze_dataset.reshape(-1, 2)
            #     sliced_imu_dataset = sliced_imu_dataset.reshape(-1, 6)
            #     for index, val in enumerate(sliced_gaze_dataset):
            #         check = np.isnan(val)
            #         if check.any():
            #             nan_index.append(index)
            #     fig = plt.figure()
            #     for i in range(len(sliced_gaze_dataset)):
            #         if i in nan_index:
            #             plt.axvline(x=i, color='r', linestyle='-', linewidth=0.1)
            #
            #     x = np.arange(len(sliced_imu_dataset))
            #     labels = ['a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z']
            #     plt.plot(x, sliced_imu_dataset[:,0], label = labels[0])
            #     plt.plot(x, sliced_imu_dataset[:,1], label = labels[1])
            #     plt.plot(x, sliced_imu_dataset[:,2], label = labels[2])
            #     plt.plot(x, sliced_imu_dataset[:,3], label = labels[3])
            #     plt.plot(x, sliced_imu_dataset[:,4], label = labels[4])
            #     plt.plot(x, sliced_imu_dataset[:,5], label = labels[5])
            #     plt.legend()
            #     fig.set_size_inches(25, 15)
            #     plt.savefig('/home/sans/Downloads/gaze_data/data_plots/' + subDir[:-1] + '_fixations.png')
                # plt.show()

            pipeline.gaze_start_index = pipeline.gaze_end_index
            pipeline.imu_start_index = pipeline.imu_end_index
            break
