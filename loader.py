import os
import json
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
from pathlib import Path
sys.path.append('../')
from variables import Variables
import cv2
from itertools import repeat

class JSON_LOADER:
    def __init__(self, folder):
        self.var = Variables()
        self.folder = folder
        if Path(self.var.root + self.folder + 'gazedata.gz').is_file():
            _ = os.system('gunzip ' + self.var.root + self.folder + 'gazedata.gz')
            _ = os.system('gunzip ' + self.var.root + self.folder + 'imudata.gz')

        with open(self.var.root + self.folder + 'gazedata') as f:
            for jsonObj in f:
                self.var.gaze_dataList.append(json.loads(jsonObj))

        with open(self.var.root + self.folder + 'imudata') as f:
            for jsonObj in f:
                self.var.imu_dataList.append(json.loads(jsonObj))

        self.imu_start_timestamp = float(self.var.imu_dataList[0]['timestamp'])
        self.gaze_start_timestamp = float(self.var.gaze_dataList[0]['timestamp'])
        self.start_timestamp = (self.imu_start_timestamp if self.gaze_start_timestamp < self.imu_start_timestamp else self.gaze_start_timestamp) - 0.01

        self.gaze_dict = {}
        self.imu_dict = {}

    def floor(self, value):
        return math.floor(value*100)/100.0

    def get_sample_rate(self, samples):
        total_sample = 0.0
        not_consistent = 0
        curr_bin = math.floor(samples[0])
        count = 0
        sample_rate = {}
        not_cons_sample_rate = {}
        for sample in samples:
            # print(sample)
            total_sample += sample - total_sample
            if total_sample > float(curr_bin)+0.99:
                sample_rate[curr_bin] = count
                # if (count != 100):
                #     not_consistent += 1
                #     not_cons_sample_rate[curr_bin] = count
                curr_bin = math.floor(total_sample)
                count = 0
            count += 1

        sample_rate[curr_bin] = count
        # if (count != 100):
        #     not_consistent += 1
        #     not_cons_sample_rate[curr_bin] = count

        return sample_rate ##if you want all the sample rates.
        # return not_cons_sample_rate, not_consistent

    def get_average_remove_dup(self, samples, avg_ind, rm_ind):
            samples[avg_ind] = (samples[avg_ind] + samples[rm_ind]) / 2.0
            samples.pop(len(samples) - abs(rm_ind))

    def POP_GAZE_DATA(self, frame_count, return_val=False):
        ### GAZE DATA
        nT, oT = 0.0, 0.0
        checked = False
        for data in self.var.gaze_dataList:
            # nT = round(float(data['timestamp']), 2)
            nT = self.floor(float(data['timestamp']))
            try:
                if(float(data['timestamp']) > self.start_timestamp):
                    diff = round(nT - oT, 2)
                    if diff > 1.0 or diff < -0.0 or diff==0.0:
                        continue

                    if diff > 0.01 :
                        # _diff = round(self.floor(float(data['timestamp'])) - oT, 2)
                        # if _diff != 0.01 and diff > 0.01:
                        diff *= 100
                        diff = int(diff)
                        a = [round(oT + 0.01*i, 2) for i in range(1, diff)]
                        self.var.gaze_data[0].extend(repeat(np.nan, diff - 1))
                        self.var.gaze_data[1].extend(repeat(np.nan, diff - 1))
                        self.var.timestamps_gaze.extend(a)
                        self.var.n_gaze_samples += diff - 1

                    if (0.0 <= float(data['data']['gaze2d'][0]) <= 1.0) and (0.0 <= float(data['data']['gaze2d'][1]) <= 1.0):
                        self.var.gaze_data[0].append(float(data['data']['gaze2d'][0]))
                        self.var.gaze_data[1].append(float(data['data']['gaze2d'][1]))
                        # checked = True

                    else:
                        self.var.gaze_data[0].append(np.nan)
                        self.var.gaze_data[1].append(np.nan)
                        # checked = True

            except Exception as e:
                self.var.gaze_data[0].append(np.nan)
                self.var.gaze_data[1].append(np.nan)
                checked = True

            if (float(data['timestamp']) > self.start_timestamp):
                self.var.n_gaze_samples += 1
                self.var.timestamps_gaze.append(nT)
                oT = nT


        if len(self.var.gaze_data[0]) < frame_count*4:
            for i in range(len(self.var.gaze_data[0]), frame_count*4):
                self.var.gaze_data[0].append(np.nan)
                self.var.gaze_data[1].append(np.nan)

        if return_val:
            return self.get_sample_rate(self.var.timestamps_gaze)

    def POP_IMU_DATA(self, frame_count, cut_short =True, return_val=False):
        nT, oT = 0.0, 0.0

        for index, data in enumerate(self.var.imu_dataList):
            nT = self.floor(float(data['timestamp']))
            try:
                if(float(data['timestamp']) > self.start_timestamp):
                    diff = round((nT - oT), 2)

                    if diff > 0.01 :
                        diff *= 100
                        diff = int(diff)
                        a = [round(oT + 0.01*i, 2) for i in range(1, diff)]
#                        print(oT, nT, diff)
                        self.var.imu_data_acc[0].extend(repeat(np.nan, diff - 1))
                        self.var.imu_data_acc[1].extend(repeat(np.nan, diff - 1))
                        self.var.imu_data_acc[2].extend(repeat(np.nan, diff - 1))

                        self.var.imu_data_gyro[0].extend(repeat(np.nan, diff - 1))
                        self.var.imu_data_gyro[1].extend(repeat(np.nan, diff - 1))
                        self.var.imu_data_gyro[2].extend(repeat(np.nan, diff - 1))
                        self.var.timestamps_imu.extend(a)
                        self.var.n_imu_samples += diff - 1

                    self.var.imu_data_acc[0].append(float(data['data']['accelerometer'][0]))
                    self.var.imu_data_acc[1].append(float(data['data']['accelerometer'][1]) ) # + 9.80665
                    self.var.imu_data_acc[2].append(float(data['data']['accelerometer'][2]))

                    self.var.imu_data_gyro[0].append(float(data['data']['gyroscope'][0]))
                    self.var.imu_data_gyro[1].append(float(data['data']['gyroscope'][1]))
                    self.var.imu_data_gyro[2].append(float(data['data']['gyroscope'][2]))

                    self.var.timestamps_imu.append(nT)
                    if cut_short:
                        if (diff <= 0.01 and self.var.check_repeat==True):
                            self.get_average_remove_dup(self.var.imu_data_acc[0], -2, -3)
                            self.get_average_remove_dup(self.var.imu_data_acc[1], -2, -3)
                            self.get_average_remove_dup(self.var.imu_data_acc[2], -2, -3)

                            self.get_average_remove_dup(self.var.imu_data_gyro[0], -2, -3)
                            self.get_average_remove_dup(self.var.imu_data_gyro[1], -2, -3)
                            self.get_average_remove_dup(self.var.imu_data_gyro[2], -2, -3)

                            self.var.timestamps_imu.pop(len(self.var.timestamps_imu) - 3)
                            self.var.n_imu_samples -= 1
                            self.var.check_repeat = False
                        elif (diff < 0.01):
                            self.var.check_repeat = True
                        else:
                            pass

                    self.var.n_imu_samples += 1
                    oT = nT
            except Exception as e:
                pass

        if len(self.var.imu_data_acc[0]) < frame_count*4:
            for i in range(len(self.var.imu_data_acc[0]), frame_count*4):
                self.var.imu_data_acc[0].append(np.nan)
                self.var.imu_data_acc[1].append(np.nan)
                self.var.imu_data_acc[2].append(np.nan)

                self.var.imu_data_gyro[0].append(np.nan)
                self.var.imu_data_gyro[1].append(np.nan)
                self.var.imu_data_gyro[2].append(np.nan)

        if return_val:
            return self.get_sample_rate(self.var.timestamps_imu)
                #############################

if __name__ == "__main__":
    folder = sys.argv[1]
    dataset_folder = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/'

    os.chdir(dataset_folder + folder)
    capture = cv2.VideoCapture('scenevideo.mp4')
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    dataset = JSON_LOADER(folder)
    print(dataset.POP_GAZE_DATA(frame_count, return_val=True))
    print(dataset.POP_IMU_DATA(frame_count, cut_short=True, return_val=True))
    print(len(dataset.var.timestamps_imu), len(dataset.var.imu_data_acc[0]), dataset.var.n_imu_samples,frame_count*4)
    print(len(dataset.var.timestamps_gaze), len(dataset.var.gaze_data[0]), dataset.var.n_gaze_samples, frame_count)
    # print(dataset.POP_IMU_DATA(frame_count, cut_short=True, return_val=True))
# print(utils.get_sample_rate(var.timestamps_imu), len(var.timestamps_imu))
# print(utils.get_sample_rate(var.timestamps_gaze), len(var.timestamps_gaze))
# plt.subplot(221)
# # plt.stem(var.timestamps_imu, var.imu_data_acc[0], 'r')
# plt.plot(var.timestamps_imu, var.imu_data_acc[0], label='x-axis')
# # s = np.array(var.imu_data_acc[1])
# # plt.specgram(var.imu_data_acc[1], Fs = 1)
# # plt.title('matplotlib.pyplot.specgram() Example\n',
# #           fontsize = 14, fontweight ='bold')
# # plt.plot(var.timestamps_imu, var.imu_data_acc[1], label='y-axis')
# # plt.plot(var.timestamps_imu, var.imu_data_acc[2], label='z-axis')
# # plt.plot(var.timestamps_imu, roll)
# plt.legend()
#
# plt.subplot(222)
# plt.plot(var.timestamps_imu, var.imu_data_gyro[0], label='x-axis')
# # plt.plot(var.timestamps_imu, var.imu_data_gyro[1], label='y-axis')
# # plt.plot(var.timestamps_imu, var.imu_data_gyro[2], label='z-axis')
# # plt.plot(var.timestamps_imu, pitch)
# plt.legend()
#
# # print('Total samples: {}, y[0]: {}, y[1]: {}'.format(len(x), len(y[0]), len(y[1])))
# plt.subplot(223)
# plt.plot(var.timestamps_gaze, var.gaze_data[0])
#
# plt.subplot(224)
# plt.plot(var.timestamps_gaze, var.gaze_data[1])
#
# plt.show()
