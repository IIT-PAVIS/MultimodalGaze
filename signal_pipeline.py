import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split
import argparse
from tqdm import tqdm
sys.path.append('../')
from variables import RootVariables
from helpers import Helpers
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from scipy.signal import butter, lfilter, freqz

class SIG_FINAL_DATASET(Dataset):
    def __init__(self, feat, labels):
        self.var = RootVariables()
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

        self.imu_data = self.standarization(self.imu_data)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def standarization(self, datas):
        datas = np.array(datas)
        seq = datas.shape[1]
        datas = datas.reshape(-1, datas.shape[-1])
        rows, cols = datas.shape
        for i in range(cols):
            mean = np.mean(datas[:,i])
            std = np.std(datas[:,i])
            datas[:,i] = (datas[:,i] - mean) / std

        datas = datas.reshape(-1, seq, datas.shape[-1])
        return datas

    def __len__(self):
        return len(self.gaze_data) # len(self.labels)

    def __getitem__(self, index):
        targets = self.gaze_data[index]
        targets[:,0] *= 512.0
        targets[:,1] *= 384.0

        return torch.from_numpy(self.imu_data[index]).to(self.device), torch.from_numpy(targets).to(self.device)


class IMU_PIPELINE(nn.Module):
    def __init__(self):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.55, bidirectional=True).to(self.device)
        # self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.dropout = nn.Dropout(0.45)
        self.activation = nn.Sigmoid()

        self.tensorboard_folder = 'signal_Adam1' #'BLSTM_signal_outputs_sell1/'

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0] - label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()
        # return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def forward(self, x):
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(out[:,-1,:]))
        return out

    def get_original_coordinates(self, pred, labels):
        # pred[:,0] *= 3.75*1920.0
        # pred[:,1] *= 2.8125*1080.0
        #
        # labels[:,0] *= 3.75*1920.0
        # labels[:,1] *= 2.8125*1080.0

        pred[:,0] *= 3.75
        pred[:,1] *= 2.8125

        labels[:,0] *= 3.75
        labels[:,1] *= 2.8125

        return pred, labels

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    var = RootVariables()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sepoch", type=int, default=0)
    # parser.add_argument('--sepoch', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--nepoch", type=int, default=15)
    parser.add_argument("--tfolder", action='store', help='tensorboard_folder name')
    args = parser.parse_args()

    # test_folder = 'test_InTheDeak_S2'
    lastFolder, newFolder = None, None
    for index, subDir in enumerate(sorted(os.listdir(var.root))):
#        if 'train_BookShelf' in subDir or 'train_CoffeeVendingMachine_S1' in subDir or 'train_CoffeeVendingMachine_S2' in subDir or 'train_CoffeeVendingMachine_S3' in subDir or 'train_InTheDeak_S1' in subDir or 'train_InTheDeak_S2' in subDir or 'train_Lift_S1' in subDir or 'train_NespressoCoffeeMachine_S1' in subDir or 'train_NespressoCoffeeMachine_S2' in subDir or 'train_Outdoor_S1' in subDir or 'train_PosterSession_S1' in subDir or 'train_PosterSession_S2' in subDir:
#            continue
        if 'train_BookShelf_S1' in subDir:
            continue
        print(subDir)
        if 'train_' in subDir:
            newFolder = subDir
            os.chdir(var.root)

            test_folder = 'test_' + newFolder[6:]
            _ = os.system('mv ' + newFolder + ' test_' + newFolder[6:])
            if lastFolder is not None:
                print('Last folder changed')
                _ = os.system('mv test_' + lastFolder[6:] + ' ' + lastFolder)

            print(newFolder, lastFolder)
            model_checkpoint = 'signal_checkpointAdam9CNN_' + test_folder[5:] + '.pth'
            # flownet_checkpoint = 'FlowNet2-SD_checkpoint.pth.tar'

            arg = 'del'
            trim_frame_size = 150
            pipeline = IMU_PIPELINE()
            pipeline.tensorboard_folder = args.tfolder
            print(pipeline)
            optimizer = optim.Adam(pipeline.parameters(), lr=0.0015, amsgrad=True) #, momentum=0.9)
            lambda1 = lambda epoch: 0.95 ** epoch
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
            criterion = nn.SmoothL1Loss()
            best_test_loss = 1000.0
            if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
                checkpoint = torch.load(pipeline.var.root + model_checkpoint)
                pipeline.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_test_loss = checkpoint['best_test_loss']
                # pipeline.current_loss = checkpoint['loss']
                print('Model loaded')

            utils = Helpers(test_folder, reset_dataset=0)
            imu_training, imu_testing, training_target, testing_target = utils.load_datasets()
            os.chdir(pipeline.var.root)
            print(torch.cuda.device_count())

            for epoch in tqdm(range(args.sepoch, args.nepochs), desc="epochs"):
                if epoch > 0:
                    utils = Helpers(test_folder, reset_dataset=0)
                    imu_training, imu_testing, training_target, testing_target = utils.load_datasets()

                trainDataset = SIG_FINAL_DATASET(imu_training_feat, training_target)
                trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
                tqdm_trainLoader = tqdm(trainLoader)
                testDataset = SIG_FINAL_DATASET(imu_testing_feat, testing_target)
                testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
                tqdm_testLoader = tqdm(testLoader)

                num_samples = 0
                total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                if epoch == 0 and 'del' in arg:
                    # _ = os.system('mv runs new_backup')
                    _ = os.system('rm -rf ' + pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)

                trainPD, testPD = [], []
                pipeline.train()
                tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
                for batch_index, (feat, labels) in enumerate(tqdm_trainLoader):
                    num_samples += feat.size(0)
                    labels = labels[:,0,:]
                    pred = pipeline(feat.float()).to(device)
                    loss = criterion(pred, labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        pred, labels = pipeline.get_original_coordinates(pred, labels)

                        # dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
                        # if batch_index > 0:
                        #     trainPD = torch.cat((trainPD, dist), 0)
                        # else:
                        #     trainPD = dist

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        total_accuracy = total_correct / num_samples
                        tqdm_trainLoader.set_description('training: ' + '_loss: {:.4} correct: {} accuracy: {:.3} MPD: {}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(trainPD)))

                pipeline.eval()
                with torch.no_grad():
                    tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Train Loss", np.mean(total_loss), epoch)
                    tb.add_scalar("Train Accuracy", total_accuracy, epoch)

                    num_samples = 0
                    total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                    dummy_correct, dummy_accuracy = 0.0, 0.0
                    for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
                        num_samples += feat.size(0)
                        labels = labels[:,0,:]
                        dummy_pts = (torch.ones(8, 2) * 0.5).to(device)
                        dummy_pts[:,0] *= 1920
                        dummy_pts[:,1] *= 1080

                        pred = pipeline(feat.float()).to(device)
                        loss = criterion(pred, labels.float())

                        # pred, labels = pipeline.get_original_coordinates(pred, labels)
                        # dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
                        # if batch_index > 0:
                        #     testPD = torch.cat((testPD, dist), 0)
                        # else:
                        #     testPD = dist

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        dummy_correct += pipeline.get_num_correct(dummy_pts.float(), labels.float())
                        dummy_accuracy = dummy_correct / num_samples
                        total_accuracy = total_correct / num_samples
                        tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} MPD: {} DAcc: {:.4}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(testPD), np.floor(100.0*dummy_accuracy)))

                tb.add_scalar("Testing Loss", np.mean(total_loss), epoch)
                tb.add_scalar("Testing Accuracy", total_accuracy, epoch)
                tb.add_scalar("Dummy Accuracy", np.floor(100.0*dummy_accuracy), epoch)
                tb.close()

                if np.mean(total_loss) <= best_test_loss:
                    best_test_loss = np.mean(total_loss)
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipeline.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_test_loss': best_test_loss,
                                }, pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
                    print('Model saved')
