import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
sys.path.append('../')
# from FlowNetPytorch.models import FlowNetS
# from flownet2.networks import FlowNetSD
from variables import RootVariables
from pytorchi3d.pytorch_i3d import Unit3D, InceptionI3d
#from skimage.transform import rotate

class Flownet_PIPELINE(nn.Module):
    def __init__(self, trim_frame_size=150, input_channels=6, batch_norm=False):
        super(VISION_PIPELINE, self).__init__()
        self.var = RootVariables()
        torch.manual_seed(1)
        self.net = FlowNetS.FlowNetS(batch_norm)
        checkpoint_path = None
        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:9]).to("cuda:0")
        for i in range(len(self.net) - 1):
             self.net[i][1] = nn.ReLU()
        self.fc1 = nn.Linear(1024*6*8, 4096).to("cuda:0")
        self.fc2 = nn.Linear(4096,256).to("cuda:0")
        self.fc3 = nn.Linear(256, 2).to("cuda:0")
        self.dropout1 = nn.Dropout(0.35)
        # self.dropout2 = nn.Dropout(0.10)
        self.activation = nn.Sigmoid()
        # self.net[8][1] = nn.ReLU(inplace=False)
        self.net[8] = self.net[8][0]
        self.tensorboard_folder = ''
        for params in self.model.parameters():
            params.requires_grad = True

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def forward(self, input_img):
        out = self.net(input_img).to("cuda:0")
#        print(out.shape)
        out = out.reshape(-1, 1024*6*8)
        out = F.relu(self.dropout(self.fc1(out))).to("cuda:0")
        out = F.relu(self.dropout(self.fc2(out))).to("cuda:0")
        out = F.relu(self.fc3(out)).to("cuda:0")

        # for index, val in enumerate(out):
        #     if out[index][0] > 512.0:
        #         out[index][0] = 512.0
        #     if out[index][1] > 384.0:
        #         out[index][1] = 384.0

        return out

    def get_original_coordinates(self, pred, labels):
        pred[:,0] *= 3.75
        pred[:,1] *= 2.8125 # 3.75

        labels[:,0] *= 3.75
        labels[:,1] *= 2.8125 # 3.75

        return pred, labels

class VISION_PIPELINE(nn.Module):
    def __init__(self, trim_frame_size=150, input_channels=6, batch_norm=False):
        super(VISION_PIPELINE, self).__init__()
        self.var = RootVariables()
        torch.manual_seed(1)
        self.dropout1 = nn.Dropout(0.25)
        self.tensorboard_folder = ''
        self.model = InceptionI3d().to("cuda:0")
        new_checkpoint = torch.load('/home/sanketthakur/Documents/gaze_pred/IMU-data_processing/pytorchi3d/models/rgb_imagenet.pt')
        self.model.load_state_dict(new_checkpoint)
        for params in self.model.parameters():
            params.requires_grad = True

        self.fc1 = nn.Linear(400, 128).to("cuda:0")
        self.fc2 = nn.Linear(128, 2).to("cuda:0")

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def forward(self, input_img):

        out = self.model(input_img)
        out = F.relu(self.dropout1(self.fc1(out)))
        out = F.relu(self.fc2(out))

        return out

    def get_original_coordinates(self, pred, labels):
        pred[:,0] *= 3.75
        pred[:,1] *= 3.75

        labels[:,0] *= 3.75
        labels[:,1] *= 3.75

        return pred, labels

class IMU_PIPELINE(nn.Module):
    def __init__(self):
        super(IMU_PIPELINE, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.55, bidirectional=True).to(self.device)
        self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)
        self.dropout = nn.Dropout(0.15)
        self.activation = nn.Sigmoid()

        self.tensorboard_folder = '' #'BLSTM_signal_outputs_sell1/'

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

        x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        out = F.relu(self.fc1(self.dropout(out[:,-1,:])))
        return out

    def get_original_coordinates(self, pred, labels):
        pred[:,0] *= 3.75
        pred[:,1] *= 3.75 # 2.8125 # 3.75

        labels[:,0] *= 3.75
        labels[:,1] *= 3.75 # 2.8125 # 3.75

        return pred, labels

class FusionPipeline(nn.Module):
    def __init__(self, test_folder, device=None):
        super(FusionPipeline, self).__init__()
        torch.manual_seed(2)
        self.device = device
        self.var = RootVariables()
        # self.checkpoint_path = self.var.root + checkpoint
        self.activation = nn.Sigmoid()
        self.temporalSeq = 32
        self.temporalSize = 16
        self.trim_frame_size = 150
        self.imuCheckpoint_file = 'signal_checkpointAdam64H_' + test_folder[5:] + '.pth'
        self.frameCheckpoint_file = 'vision_checkpointAdami3d_' + test_folder[5:] +'.pth'

        ## IMU Models
        self.imuModel = IMU_ENCODER()
        imuCheckpoint = torch.load(self.var.root + 'datasets/' + test_folder[5:] + '/' + self.imuCheckpoint_file,  map_location="cuda:0")
        self.imuModel.load_state_dict(imuCheckpoint['model_state_dict'])
        for params in self.imuModel.parameters():
             params.requires_grad = False

        ## FRAME MODELS
        self.frameModel =  i3d_VIS_ENCODER()
        frameCheckpoint = torch.load(self.var.root + 'datasets/' + test_folder[5:] + '/' + self.frameCheckpoint_file,  map_location="cuda:0")
        self.frameModel.load_state_dict(frameCheckpoint['model_state_dict'])
        for params in self.frameModel.parameters():
            params.requires_grad = False

        ## TEMPORAL MODELS
 #       self.temporalModel = TEMP_ENCODER(self.temporalSize)

        self.dropout = nn.Dropout(0.2)
        # self.imu_down_fc = nn.Linear(256, 128).to("cuda:0")

        self.fc0 = nn.Linear(256, 128).to("cuda:0")
        self.fc1 = nn.Linear(128, 2).to("cuda:0")
        # self.fc0 = nn.Linear(512, 256).to("cuda:0")
        # self.fc1 = nn.Linear(256, 2).to("cuda:0")
        ##OTHER
        self.imu_encoder_params = None
        self.frame_encoder_params = None
        self.imuBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to("cuda:0")
        self.frameBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to("cuda:0")
        # self.imuBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to("cuda:0")
        # self.frameBN = nn.BatchNorm1d(self.var.hidden_size*2, affine=True).to("cuda:0")
        self.concatBN = nn.BatchNorm1d(256).to("cuda:0")
        self.fcBN = nn.BatchNorm1d(128).to("cuda:0")    ## change to 256 when using flownet
        self.tensorboard_folder = ''

    def get_encoder_params(self, imu_BatchData, frame_BatchData):
        # imu_BatchData = self.imu_down_fc(imu_BatchData).to("cuda:0")
        self.imu_encoder_params = self.imuBN(F.relu(self.imuModel(imu_BatchData.float()))).to("cuda:0")
        self.frame_encoder_params = self.frameBN(F.relu(self.frameModel(frame_BatchData.float()))).to("cuda:0")
        return self.imu_encoder_params, self.frame_encoder_params

    def fusion_network(self, imu_params, frame_params):
        return torch.cat((frame_params, imu_params), dim=1).to("cuda:0")

    def temporal_modelling(self, fused_params):
 #       newParams = fused_params.reshape(fused_params.shape[0], self.temporalSeq, self.temporalSize)
 #       tempOut = self.temporalModel(newParams.float()).to("cuda:2")
 #       gaze_pred = self.fc1(tempOut).to("cuda:2")
 #       print(fused_params, self.fc0.weight)
        # fused_params = self.concatBN(fused_params).to("cuda:0")
        gaze_pred = self.fcBN(F.relu(self.fc0(self.dropout(fused_params)))).to("cuda:0")
        gaze_pred = F.relu(self.fc1(gaze_pred)).to("cuda:0")
#        gaze_pred = self.fc2(self.dropout(gaze_pred)).to("cuda:2")

        return gaze_pred

    def forward(self, batch_frame_data, batch_imu_data):
        imu_params, frame_params = self.get_encoder_params(batch_imu_data, batch_frame_data)
        fused = self.fusion_network(imu_params, frame_params)
        coordinate = self.temporal_modelling(fused)

        # for index, val in enumerate(coordinate):
        #     if coordinate[index][0] > 512.0:
        #         coordinate[index][0] = 512.0
        #     if coordinate[index][1] > 384.0:
        #         coordinate[index][1] = 384.0

        return coordinate

    def get_num_correct(self, pred, label):
        return torch.logical_and((torch.abs(pred[:,0]-label[:,0]) <= 100.0), (torch.abs(pred[:,1]-label[:,1]) <= 100.0)).sum().item()

    def get_original_coordinates(self, pred, labels):
        pred[:,0] *= 3.75
        pred[:,1] *= 3.75 #2.8125

        labels[:,0] *= 3.75
        labels[:,1] *= 3.75

        return pred, labels

class IMU_ENCODER(nn.Module):
    def __init__(self):
        super(IMU_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(self.var.imu_input_size, self.var.hidden_size, self.var.num_layers, batch_first=True, dropout=0.65, bidirectional=True).to("cuda:0")
        # self.fc0 = nn.Linear(6, self.var.imu_input_size).to(self.device)
        self.fc1 = nn.Linear(self.var.hidden_size*2, 2).to(self.device)

    def forward(self, x):
        h0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size, requires_grad=True).to(self.device)
        # h0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)
        # c0 = torch.zeros(self.var.num_layers*2, self.var.batch_size, self.var.hidden_size).to(self.device)

        # x = self.fc0(x)
        out, _ = self.lstm(x, (h0, c0))
        return out[:,-1,:]

class TEMP_ENCODER(nn.Module):
    def __init__(self, input_size):
        super(TEMP_ENCODER, self).__init__()
        torch.manual_seed(0)
        self.var = RootVariables()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(input_size, int(self.var.hidden_size/2), int(self.var.num_layers/2), batch_first=True, dropout=0.45, bidirectional=True).to("cuda:0")

    def forward(self, x):
        # hidden = (h0, c0)
        h0 = torch.randn(self.var.num_layers, self.var.batch_size, int(self.var.hidden_size/2), requires_grad=True).to("cuda:0")
        c0 = torch.randn(self.var.num_layers, self.var.batch_size, int(self.var.hidden_size/2), requires_grad=True).to("cuda:0")
        out, _ = self.lstm(x, (h0, c0))
        # out = self.activation(self.fc1(out[:,-1,:]))
        return out[:,-1,:]

class i3d_VIS_ENCODER(nn.Module):
    def __init__(self, trim_frame_size=150, input_channels=6, batch_norm=False):
        super(i3d_VIS_ENCODER, self).__init__()
        self.var = RootVariables()
        torch.manual_seed(1)
        self.dropout1 = nn.Dropout(0.25)
        self.model = InceptionI3d().to("cuda:0")
        new_checkpoint = torch.load('/home/sanketthakur/Documents/gaze_pred/IMU-data_processing/pytorchi3d/models/rgb_imagenet.pt')
        self.model.load_state_dict(new_checkpoint)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(400, 128).to("cuda:0")
        self.fc2 = nn.Linear(128, 2).to("cuda:0")


    def forward(self, input_img):
        out = self.model(input_img)
        out = F.relu(self.dropout(self.fc1(out)))
        return out

class VIS_ENCODER(nn.Module):
    def __init__(self, checkpoint_path, input_channels=6, batch_norm=False):
        super(VIS_ENCODER, self).__init__()

        self.var = RootVariables()
        torch.manual_seed(1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = FlowNetS.FlowNetS(batch_norm)
        dict = torch.load(checkpoint_path)
        self.net.load_state_dict(dict["state_dict"])
        self.net = nn.Sequential(*list(self.net.children())[0:9]).to(self.device)
        for i in range(len(self.net) - 1):
            self.net[i][1] = nn.ReLU()

        self.fc1 = nn.Linear(1024*6*8, 256).to(self.device)
#        self.fc2 = nn.Linear(4096, 256).to(self.device)
        self.fc3 = nn.Linear(256, 2).to(self.device)
        self.dropout = nn.Dropout(0.3)
        # self.net[8][1] = nn.ReLU(inplace=False)
        self.net[8] = self.net[8][0]

#        for params in self.net.parameters():
#            params.requires_grad = True

    def forward(self, input_img):
        out = self.net(input_img)
        out = out.reshape(-1, 1024*6*8)
        out = F.relu(self.dropout(self.fc1(out)))
        #out = F.leaky_relu(self.dropout(self.fc2(out)), 0.1)
        # out = self.activation(self.fc3(out))

        return out
