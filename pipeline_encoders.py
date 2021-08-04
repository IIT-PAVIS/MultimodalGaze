import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys, os, ast
sys.path.append('../')
from variables import RootVariables
from FlowNetPytorch.models import FlowNetS

device = torch.device("cpu")



## PREPARING THE DATA
# folder = sys.argv[1]
# dataset_folder = '/home/sans/Downloads/gaze_data/'
# os.chdir(dataset_folder + folder + '/' if folder[-1]!='/' else (dataset_folder + folder))
if __name__ == "__main__":
    folder = sys.argv[1]
    device = torch.device("cpu")

    var = RootVariables()
    os.chdir(var.root + folder)
    # dataset = FRAME_IMU_DATASET(var.root, folder, 150, device)
    # trainLoader = torch.utils.data.DataLoader(dataset, batch_size=var.batch_size, drop_last=True)
    # a = iter(trainLoader)
    # f, g, i = next(a)
    # # print(data.shape, data)
    # print(i.shape) # [batch_size, sequence_length, input_size]
    # i = i.reshape(i.shape[0], i.shape[2], -1)
    # print(i.shape)

    model = IMU_ENCODER(var.imu_input_size ,device).to(device)
    imuCheckpoint_file = 'hidden_256_60e_signal_pipeline_checkpoint.pth'
    imuCheckpoint = torch.load(var.root + imuCheckpoint_file)
    model.load_state_dict(imuCheckpoint['model_state_dict'])
    print(model)
    # scores = model(data.float())
    # print(model, scores.shape)
    # scores = scores.unsqueeze(dim = 1)
    # newscore = scores.reshape(scores.shape[0], 4, 32)
    # print(newscore.shape)
    # print(newscore)
