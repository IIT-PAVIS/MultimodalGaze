import sys, os, cv2
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')
from prepare_dataset import IMU_GAZE_FRAME_DATASET
from loader import JSON_LOADER
from variables import RootVariables

frame_size = 256
trim_size = 150
var = RootVariables()
def standarization(datas):
    datas = datas.reshape(-1, datas.shape[-1])
    print(datas.shape)
    dummy = np.zeros(datas.shape)
    rows, cols = datas.shape
    for i in range(cols):
        mean = np.mean(datas[:,i])
        std = np.std(datas[:,i])
        dummy[:,i] = (datas[:,i] - mean) / std
    return dummy

def normalization(datas):
    datas = datas.reshape(-1, datas.shape[-1])
    print(datas.shape)
    dummy = np.zeros(datas.shape)
    rows, cols = datas.shape
    for i in range(cols):
        max = np.max(datas[:,i])
        min = np.min(datas[:,i])
        dummy[:,i] = (datas[:,i] - min ) / (max - min)
    return dummy

datasets = IMU_GAZE_FRAME_DATASET(var.root, frame_size, trim_size)
uni_imu_dataset = datasets.imu_datasets
n_uni_imu_dataset = normalization(uni_imu_dataset)
s_uni_imu_dataset = standarization(uni_imu_dataset)
start_index, end_index = 0, 0
for index, subDir in enumerate(sorted(os.listdir(var.root))):
    if 'imu_Book' in subDir:
        subDir  = subDir + '/' if subDir[-1]!='/' else  subDir
        os.chdir(var.root + subDir)
        capture = cv2.VideoCapture('scenevideo.mp4')
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        dataset = JSON_LOADER(subDir)
        imu = dataset.POP_IMU_DATA(frame_count, return_val=True)
        imu_arr_acc = np.array(dataset.var.imu_data_acc).transpose()
        imu_arr_gyro = np.array(dataset.var.imu_data_gyro).transpose()
        dataset = JSON_LOADER(subDir)
        new_imu = dataset.POP_IMU_DATA(frame_count, cut_short=False, return_val=True)
        acc_copy = np.array(dataset.var.imu_data_acc).transpose()
        gyro_copy = np.array(dataset.var.imu_data_gyro).transpose()
        end_index = start_index + len(acc_copy)
        print(subDir, start_index, end_index)
        n_acc = n_uni_imu_dataset[start_index:end_index, :3]
        n_gyro = n_uni_imu_dataset[start_index:end_index, 3:]
        s_acc = s_uni_imu_dataset[start_index:end_index, :3]
        s_gyro = s_uni_imu_dataset[start_index:end_index, 3:]
        start_index = end_index

        x = np.arange(0, len(imu_arr_acc))
        new_x = np.arange(0, len(acc_copy))
        fig, ax = plt.subplots(nrows=4, ncols=2)
        for r_index, row in enumerate(ax):
            ax = [x, new_x, new_x, new_x]
            data = [[imu_arr_acc, imu_arr_gyro], [acc_copy, gyro_copy], [s_acc, s_gyro], [n_acc, n_gyro]]
            for c_index, col in enumerate(row):
                col.plot(ax[r_index], data[r_index][c_index])

        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        fig.set_size_inches(20, 15)
        # plt.savefig('/home/sans/Downloads/gaze_data/data_plots/' + subDir[:-1] + '.png')
        plt.show()
        # fig = plt.figure()
        # fig.add_subplot(221)
        # plt.plot(x, imu_arr_acc)
        # fig.add_subplot(222)
        # plt.plot(x, imu_arr_gyro)
        # new_x = np.arange(0, len(acc_copy))
        # fig.add_subplot(223)
        # plt.plot(new_x, acc_copy)
        # fig.add_subplot(224)
        # plt.plot(new_x, gyro_copy)
        # plt.show()
