import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import argparse
import cv2, sys, os
from flownet2.networks import FlowNetS
sys.path.append('../')
# from getDataset import ImageDataset
from variables import RootVariables

class VIS_ENCODER(nn.Module):
    def __init__(self, args, checkpoint_path, device, input_channels=6, batch_norm=False):
        super(VIS_ENCODER, self).__init__()

        self.var = RootVariables()
        torch.manual_seed(1)
        self.device = device
        self.net = FlowNetS.FlowNetS(args, input_channels, batch_norm)
        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:9]).to(self.device)
        for i in range(len(self.net) - 1):
            self.net[i][1] = nn.ReLU()

        self.fc1 = nn.Linear(1024*4*4, 4096).to(self.device)
        self.fc2 = nn.Linear(4096, 256).to(self.device)
        self.fc3 = nn.Linear(256, 2).to(self.device)
        self.dropout = nn.Dropout(0.35)
        self.activation = nn.Sigmoid()
        # self.net[8][1] = nn.ReLU(inplace=False)
        self.net[8] = self.net[8][0]

        for params in self.net.parameters():
            params.requires_grad = True

    def forward(self, input_img):
        out = self.net(input_img)
        out = out.reshape(-1, 1024*4*4)
        out = F.relu(self.dropout(self.fc1(out)))
        out = F.relu(self.dropout(self.fc2(out)))
        # out = self.activation(self.fc3(out))

        return out


if __name__ == "__main__":
    var = RootVariables()
    device = torch.device("cpu")
    folder = 'imu_BookShelf_S1/'
    os.chdir(var.root + folder)

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)

    args = parser.parse_args()
    ## load model without batch norm
    checkpoint_path = var.root + "FlowNet2-S_checkpoint.pth.tar"

    model = VIS_ENCODER(args, checkpoint_path, device)
    imuCheckpoint_file = 'hidden_256_55e_vision_pipeline_checkpoint.pth'
    imuCheckpoint = torch.load(var.root + imuCheckpoint_file)
    model.load_state_dict(imuCheckpoint['model_state_dict'])
    net = nn.Sequential(*list(model.children())[0:3])
    print(net)
    # print(img_enc)
    # output = img_enc.run_model(imgs)
