import os, cv2
from tqdm import tqdm
import torch, argparse
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pipeline_new import FusionPipeline, FINAL_DATASET
from helpers import Helpers
from variables import RootVariables

if __name__ == "__main__":

    var = RootVariables()
    folder = 'test_smallGroupMeeting_S3'
    utils = Helpers(folder, reset_dataset=0)
    _, imu_testing_feat, _, targets = utils.load_datasets()

    flownet_checkpoint = 'flownets_EPE1.951.pth.tar'
    pipeline = FusionPipeline(flownet_checkpoint, folder)
    # print(pipeline)


    model_checkpoint = 'pipeline_checkpointAdam_' + folder[5:] + '.pth'
    if Path(pipeline.var.root + 'datasets/' + folder[5:] + '/' + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + 'datasets/' + folder[5:] + '/' + model_checkpoint,  map_location="cuda:0")
        pipeline.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded')

    pipeline.eval()
    testPD, dummyPD = None, None
    catList = []

    with torch.no_grad():
        testDataset = FINAL_DATASET('testing_images', imu_testing_feat, targets)
        testLoader = torch.utils.data.DataLoader(testDataset, shuffle=False, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)

#        tqdm_trainLoader = tqdm(trainLoader)
        tqdm_testLoader = tqdm(testLoader)

        num_samples = 0
        total_loss, total_correct, total_accuracy = [], 0.0, 0.0
        dummy_correct, dummy_accuracy = 0.0, 0.0
        for batch_index, (frame_feat, imu_feat, labels) in enumerate(tqdm_testLoader):

            num_samples += frame_feat.size(0)
            labels = labels[:,0,:]
            dummy_pts = (torch.ones(8, 2) * 0.5).to("cuda:0")
            dummy_pts[:,0] *= 1920
            dummy_pts[:,1] *= 1080

            pred = pipeline(frame_feat, imu_feat).float()
            pred, labels = pipeline.get_original_coordinates(pred, labels)

            dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
            pred = pred.unsqueeze(dim=0)
            catList.append(pred)

    os.chdir(var.root + folder)
    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES,var.trim_frame_size)
    ret, frame = capture.read()
    print(catList[0])

    for i in range(frame_count - 30000):
        if ret == True:
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 512, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            try:
                gt_gaze_pts = targets[i][0]

                # gt_gaze_pts = np.sum(sliced_gaze_dataset[i], axis=0) / 4.0
                # pred_gaze_pts = coordinate[i]
                padding_r = 100.0
                padding = 100.0
                # plt.scatter(int(pred_gaze_pts[0]*frame.shape[1]), int(pred_gaze_pts[1]*frame.shape[0]))

                start_point = (int(gt_gaze_pts[0]*frame.shape[1]) - int(padding), int(gt_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                end_point = (int(gt_gaze_pts[0]*frame.shape[1]) + int(padding), int(gt_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                # pred_start_point = (int(pred_gaze_pts[0]*frame.shape[1]) - int(padding), int(pred_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                # pred_end_point = (int(pred_gaze_pts[0]*frame.shape[1]) + int(padding), int(pred_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                #
                frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
                # frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)

                frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]),int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
                frame = cv2.circle(frame, (int(catList[0]),int(catList[1])), radius=5, color=(0, 0, 255), thickness=5)
                # frame = cv2.circle(frame, (int(pred_gaze_pts[0]*frame.shape[1]),int(pred_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 255, 0), thickness=5)
            except Exception as e:
                print(e)
            cv2.imshow('image', frame)
            # out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(0)
            ret, frame = capture.read()
