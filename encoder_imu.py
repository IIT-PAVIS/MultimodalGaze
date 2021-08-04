import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys, os, ast
sys.path.append('../')
from variables import RootVariables

device = torch.device("cpu")

class IMU_ENCODER(nn.Module):
    def __init__(self, input_size, device):
        super(IMU_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = device
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.65, bidirectional=True).to(self.device)
        # self.fc0 = nn.Linear(6, self.var.imu_input_size)

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.activation(self.fc1(out[:,-1,:]))
        return out[:,-1,:]

class TEMP_ENCODER(nn.Module):
    def __init__(self, input_size, device):
        super(TEMP_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = device
        self.lstm = nn.LSTM(input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, bidirectional=True).to(self.device)
        self.fc0 = nn.Linear(6, self.var.imu_input_size)

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.activation(self.fc1(out[:,-1,:]))
        return out[:,-1,:]

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
