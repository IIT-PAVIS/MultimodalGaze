import sys, os
import numpy as np
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
from tqdm import tqdm
sys.path.append('../')
from pipeline_encoders import IMU_ENCODER, TEMP_ENCODER, VIS_ENCODER
from helpers import Helpers
# from FlowNetPytorch.models import FlowNetS
from variables import RootVariables
from torch.utils.tensorboard import SummaryWriter

class FusionPipeline(nn.Module):
    def __init__(self):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.var = RootVariables()
        self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 32
        self.temporalSize = 16
        self.trim_frame_size = 150
        self.imuCheckpoint_file = 'signal_checkpoint0_' + test_folder[5:] + '.pth'
        self.frameCheckpoint_file = 'vision_checkpointAdam9CNN_' + test_folder[5:] +'.pth'

        ## IMU Models
        self.imuModel = IMU_ENCODER()
        imuCheckpoint = torch.load(self.var.root + 'datasets/' + self.var.test_folder[5:] + '/' + self.imuCheckpoint_file,  map_location="cuda:0")
        self.imuModel.load_state_dict(imuCheckpoint['model_state_dict'])
        for params in self.imuModel.parameters():
             params.requires_grad = True

        ## FRAME MODELS
        self.frameModel =  VIS_ENCODER()
        frameCheckpoint = torch.load(self.var.root + 'datasets/' + self.var.test_folder[5:] + '/' + self.frameCheckpoint_file,  map_location="cuda:0")
        self.frameModel.load_state_dict(frameCheckpoint['model_state_dict'])
        for params in self.frameModel.parameters():
            params.requires_grad = True

        ## TEMPORAL MODELS
 #       self.temporalModel = TEMP_ENCODER(self.temporalSize)

#        self.fc1 = nn.Linear(self.var.hidden_size, 2).to("cuda:2")
        self.dropout = nn.Dropout(0.35)
        self.fc0 = nn.Linear(512, 256).to("cuda:0")
        self.fc1 = nn.Linear(256, 2).to("cuda:0")
#        self.fc2 = nn.Linear(128, 2).to("cuda:2")
        ##OTHER
        self.imu_encoder_params = None
        self.frame_encoder_params = None
        self.imuBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to("cuda:0")
        self.frameBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to("cuda:0")
        self.fcBN = nn.BatchNorm1d(256).to("cuda:0")
        self.tensorboard_folder = ''

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        self.imu_encoder_params = F.relu(self.imuBN(self.imuModel(imu_BatchData.float()))).to("cuda:0")
        self.frame_encoder_params = F.relu(self.frameBN(self.frameModel(frame_BatchData.float()))).to("cuda:0")
#        self.frame_encoder_params = F.leaky_relu(self.dropout(self.downsample(self.frame_encoder_params)), 0.1).to("cuda:1")
        return self.imu_encoder_params, self.frame_encoder_params

    def fusion_network(self, imu_params, frame_params):
        return torch.cat((frame_params, imu_params), dim=1).to("cuda:0")

    def temporal_modelling(self, fused_params):
 #       newParams = fused_params.reshape(fused_params.shape[0], self.temporalSeq, self.temporalSize)
 #       tempOut = self.temporalModel(newParams.float()).to("cuda:2")
 #       gaze_pred = self.fc1(tempOut).to("cuda:2")
 #       print(fused_params, self.fc0.weight)
        gaze_pred = F.relu(self.fcBN(self.fc0(self.dropout(fused_params)))).to("cuda:0")
        gaze_pred = F.relu(self.fc1(self.dropout(gaze_pred))).to("cuda:0")
#        gaze_pred = self.fc2(self.dropout(gaze_pred)).to("cuda:2")

        return gaze_pred

    def forward(self, batch_frame_data, batch_imu_data):
        imu_params, frame_params = self.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = self.fusion_network(imu_params, frame_params)
        coordinate = self.temporal_modelling(fused)

        for index, val in enumerate(coordinate):
            if coordinate[index][0] > 512.0:
                coordinate[index][0] = 512.0
            if coordinate[index][1] > 384.0:
                coordinate[index][1] = 384.0

        return coordinate

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def get_original_coordinates(self, pred, labels):
        pred[:,0] *= 3.75
        pred[:,1] *= 2.8125

        labels[:,0] *= 3.75
        labels[:,1] *= 2.8125

        return pred, labels

class FINAL_DATASET(Dataset):
    def __init__(self, folder_type, imu_feat, labels):
        self.var = RootVariables()
        self.folder_type = folder_type
        self.imu_data = []
        self.indexes = []
        checkedLast = False
        for index in range(len(labels)):
            check = np.isnan(labels[index])
            imu_check = np.isnan(imu_feat[index])
            if check.any() or imu_check.any():
                continue
            else:
                self.indexes.append(index)
                self.imu_data.append(imu_feat[index])

        self.imu_data = self.standarization(self.imu_data)

        self.transforms = transforms.Compose([transforms.ToTensor()])
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return len(self.indexes) # len(self.labels)

    def __getitem__(self, index):
        f_index = self.indexes[index]
#        img = self.frames[f_index]
        img =  np.load(self.var.root + self.folder_type + '/frames_' + str(f_index) +'.npy')
        targets = self.gaze_data[f_index]
        targets[:,0] *= 512.0
        targets[:,1] *= 384.0

        return self.transforms(img).to("cuda:0"), torch.from_numpy(self.imu_data[index]).to("cuda:0"), torch.from_numpy(targets).to("cuda:0")

if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    var = RootVariables()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sepoch", type=int, default=0)
    # parser.add_argument('--sepoch', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--nepoch", type=int, default=15)
    parser.add_argument("--tfolder", action='store', help='tensorboard_folder name')
    parser.add_argument("--reset_data", type=int)
    args = parser.parse_args()

    lastFolder, newFolder = None, None
    for index, subDir in enumerate(sorted(os.listdir(var.root))):
#        if 'train_BookShelf_S1' in subDir:
#            continue
        if 'train_PosterSession' in subDir:
            print(subDir)
            newFolder = subDir
            os.chdir(var.root)

            test_folder = 'test_' + newFolder[6:]
            _ = os.system('mv ' + newFolder + ' test_' + newFolder[6:])
            if lastFolder is not None:
                print('Last folder changed')
                _ = os.system('mv test_' + lastFolder[6:] + ' ' + lastFolder)

            print(newFolder, lastFolder)
            model_checkpoint = 'pipeline_checkpointAdam_' + test_folder[5:] + '.pth'
            arg = 'del'
            trim_frame_size = 150
            var.test_folder = test_folder

            pipeline = FusionPipeline()
            pipeline.tensorboard_folder = args.tfolder
            optimizer  = optim.Adam(pipeline.parameters(), lr=0.00095,  amsgrad=True)
#            lambda1 = lambda epoch: 0.85 ** epoch
#            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=-1)
            criterion = nn.L1Loss()
            print(pipeline)
            best_test_acc = 0.0
            if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
                checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
                pipeline.load_state_dict(checkpoint['model_state_dict'])
                best_test_acc = checkpoint['best_test_acc']
                # pipeline.current_loss = checkpoint['loss']
                print('Model loaded')
            #if 'BookShelf​' in test_folder:
            #    utils = Helpers(test_folder, reset_dataset=1)
            #else:
            utils = Helpers(test_folder, reset_dataset=args.reset_data)
            n_epochs = 15
            imu_training, imu_testing, training_target, testing_target = utils.load_datasets()
            os.chdir(pipeline.var.root)

            for epoch in tqdm(range(args.sepoch, args.nepochs), desc="epochs"):
                if epoch > 0:
                    utils = Helpers(test_folder, reset_dataset=0)
                    imu_training, imu_testing, training_target, testing_target = utils.load_datasets()

#                ttesting_target = np.copy(testing_target)
#                timu_training = np.copy(imu_testing)
                trainDataset = FINAL_DATASET('training_images', imu_training, training_target)
                trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
                tqdm_trainLoader = tqdm(trainLoader)
                testDataset = FINAL_DATASET('testing_images', imu_testing,  testing_target)
                testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
                tqdm_testLoader = tqdm(testLoader)

                if epoch == 0 and 'del' in arg:
                    # _ = os.system('mv runs new_backup')
                    _ = os.system('rm -rf ' + pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)

                num_samples = 0
                total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                pipeline.train()
                tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
                for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_trainLoader):
                    num_samples += frame_feat.size(0)
                    labels = labels[:,0,:]
                    pred = pipeline(frame_feat, imu_feat)
#                    print(pred, labels)
                    loss = criterion(pred.float(), labels.float())
                    optimizer.zero_grad()
                    loss.backward()
                    ## add gradient clipping
#                    nn.utils.clip_grad_value_(pipeline.parameters(), clip_value=1.0)
                    optimizer.step()

                    with torch.no_grad():

                        pred, labels = pipeline.get_original_coordinates(pred, labels)

#                        dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
#                        if batch_index > 0:
#                            trainPD = torch.cat((trainPD, dist), 1)
#                        else:
#                            trainPD = dist

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        total_accuracy = total_correct / num_samples
                        tqdm_trainLoader.set_description('training: ' + '_loss: {:.4} correct: {} accuracy: {:.3} lr:{}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy, optimizer.param_groups[0]['lr']))

#                scheduler.step()
                pipeline.eval()
                with torch.no_grad():
                    tb = SummaryWriter(pipeline.var.root + 'datasets/' + test_folder[5:] + '/runs/' + pipeline.tensorboard_folder)
                    tb.add_scalar("Train Loss", np.mean(total_loss), epoch)
                    tb.add_scalar("Training Correct", total_correct, epoch)
                    tb.add_scalar("Train Accuracy", total_accuracy, epoch)

                    num_samples = 0
                    total_loss, total_correct, total_accuracy = [], 0.0, 0.0
                    dummy_correct, dummy_accuracy = 0.0, 0.0
                    for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_testLoader):
                        num_samples += frame_feat.size(0)
                        labels = labels[:,0,:]
                        dummy_pts = (torch.ones(8, 2) * 0.5).to("cuda:0")
                        dummy_pts[:,0] *= 1920
                        dummy_pts[:,1] *= 1080

                        pred = pipeline(frame_feat, imu_feat)
                        loss = criterion(pred.float(), labels.float())
                        pred, labels = pipeline.get_original_coordinates(pred, labels)

#                        dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
#                        if batch_index > 0:
#                            testPD = torch.cat((testPD, dist), 1)
#                        else:
#                            testPD = dist

                        total_loss.append(loss.detach().item())
                        total_correct += pipeline.get_num_correct(pred, labels.float())
                        dummy_correct += pipeline.get_num_correct(dummy_pts.float(), labels.float())
                        dummy_accuracy = dummy_correct / num_samples
                        total_accuracy = total_correct / num_samples
                        tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} DAcc: {:.4}'.format(
                            np.mean(total_loss), total_correct, 100.0*total_accuracy,  np.floor(100.0*dummy_accuracy)))

                tb.add_scalar("Testing Loss", np.mean(total_loss), epoch)
                tb.add_scalar("Testing Correct", total_correct, epoch)
                tb.add_scalar("Testing Accuracy", total_accuracy, epoch)
                tb.add_scalar("Dummy Accuracy", np.floor(100.0*dummy_accuracy), epoch)
                tb.close()

                if total_accuracy >= best_test_acc:
                    best_test_acc = total_accuracy
                    torch.save({
                                'epoch': epoch,
                                'model_state_dict': pipeline.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_test_acc': best_test_acc,
                                }, pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
                    print('Model saved')

            lastFolder = newFolder

    # optimizer = optim.Adam([
    #                         {'params': imuModel.parameters(), 'lr': 1e-4},
    #                         {'params': frameModel.parameters(), 'lr': 1e-4},
    #                         {'params': temporalModel.parameters(), 'lr': 1e-4}
    #                         ], lr=1e-3)
