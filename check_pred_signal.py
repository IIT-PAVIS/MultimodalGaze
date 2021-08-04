import os, cv2, sys
from tqdm import tqdm
import torch, argparse
import torch.nn as nn
from pathlib import Path
import numpy as np
import random
sys.path.append('../')
from signal_pipeline import IMU_PIPELINE, FINAL_DATASET
from variables import RootVariables
from helpers import Helpers

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_folder = 'test_InTheDeak_S2'
    model_checkpoint = 'signal_checkpoint0_' + test_folder[5:] + '.pth'

    pipeline = IMU_PIPELINE()
    criterion = nn.L1Loss()
    print(pipeline)
    best_test_loss = 1000.0
    trim_frame_size = 150

    if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
        checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint)
        pipeline.load_state_dict(checkpoint['model_state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_test_loss = checkpoint['best_test_loss']
        # pipeline.current_loss = checkpoint['loss']
        print('Model loaded')

    utils = Helpers(test_folder, reset_dataset=0)
    imu_training, imu_testing, training_target, testing_target = utils.load_datasets()
    os.chdir(pipeline.var.root)
    tt = np.copy(testing_target)
    print(tt[0])

    testDataset = FINAL_DATASET(imu_testing, testing_target)
    testLoader = torch.utils.data.DataLoader(testDataset, shuffle=False, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
    tqdm_testLoader = tqdm(testLoader)

    pipeline.eval()
    total_loss, total_correct, total_accuracy = [], 0.0, 0.0
    predList, labelList, testPD = None, None, None
    num_samples = 0
    for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
        num_samples += feat.size(0)
        labels = labels[:,0,:]

        pred = pipeline(feat.float()).to(device)
        loss = criterion(pred, labels.float())

        pred, labels = pipeline.get_original_coordinates(pred, labels)
        dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
#        print(pred, labels, dist)
        if batch_index > 0:
            testPD = torch.cat((testPD, dist), 1)
            predList = torch.cat((predList, pred), 0)
            labelList = torch.cat((labelList, labels), 0)
        else:
            testPD = dist
            predList = pred
            labelList = labels

        total_loss.append(loss.detach().item())
        total_correct += pipeline.get_num_correct(pred, labels.float())
        # print(total_correct, pred, labels)
        total_accuracy = total_correct / num_samples
        tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} MPD: {}'.format(
            np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(testPD)))


    os.chdir(pipeline.var.root + test_folder)
    video_file = 'scenevideo.mp4'
    capture = cv2.VideoCapture(video_file)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    # print(frame_count, fps, predList[0], testing_target[0])
    capture.set(cv2.CAP_PROP_POS_FRAMES,trim_frame_size+1)
    ret, frame = capture.read()
    # subDir = 'val_SuperMarket_S1/'
    # coordinate = torch.load(pipeline.var.root + 'signal_' + subDir[4:-1] + '_predictions.pt', map_location=torch.device('cpu'))
    # coordinate = coordinate.detach().cpu().numpy()
    # print(coordinate[0])
    # print(len(coordinate), len(sliced_gaze_dataset))

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('signal_output.mp4',fourcc, fps, (frame.shape[1],frame.shape[0]))
    # frame_count = 0
    # df_gaze = df_gaze.T
    acc = 0
    for i in range(frame_count - 30000):
        if ret == True:
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('image', 512, 512)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # coordinate = sliced_gaze_dataset[i]
            # pred_gaze_pts = coordinate[i]
            # for index, pt in enumerate(coordinate):
            #     try:
            #         (x, y) = pt[0], pt[1]
            #         frame = cv2.circle(frame, (int(x*frame.shape[1]),int(y*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
            #     except Exception as e:
            #         print(e)
            try:
                # print(tt[i], testing_target[i])
                # frame = cv2.resize(frame, (512, 384))
                # gt_gaze_pts = tt[i][0]
                pred_gaze_pts = predList[i]
                gt_gaze_pts = labelList[i]
                # gt_gaze_pts[0] *= 1920.0
                # gt_gaze_pts[1] *= 1080.0
                # gt_gaze_pts[0] *= 512.0
                # gt_gaze_pts[1] *= 384.0
                print(gt_gaze_pts, labelList[i])
                pred_gaze_pts = predList[i]
                # frame = cv2.resize(frame, (512, 384))

                padding_r = 50.0
                padding = 50.0
                sign = 1 if random.random() > 0.5 else -1
                # start_point = (int(gt_gaze_pts[0]*frame.shape[1]) - int(padding), int(gt_gaze_pts[1]*frame.shape[0]) + int(padding_r))
                # end_point = (int(gt_gaze_pts[0]*frame.shape[1]) + int(padding), int(gt_gaze_pts[1]*frame.shape[0]) - int(padding_r))
                # pred_start_point = (int(gt_gaze_pts[0]*frame.shape[1] - sign*padding) - int(padding), int(gt_gaze_pts[1]*frame.shape[0] - sign*padding_r) + int(padding_r))
                # pred_end_point = (int(gt_gaze_pts[0]*frame.shape[1] - sign*padding) + int(padding), int(gt_gaze_pts[1]*frame.shape[0] - sign*padding_r) - int(padding_r))
                # #
                # frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
                # frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)
                #
                # frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1]) ,int(gt_gaze_pts[1]*frame.shape[0])), radius=5, color=(0, 0, 255), thickness=5)
                # frame = cv2.circle(frame, (int(gt_gaze_pts[0]*frame.shape[1] - sign*padding) ,int(gt_gaze_pts[1]*frame.shape[0] - sign*padding_r)), radius=5, color=(0, 255, 0), thickness=5)

                frame = cv2.circle(frame, (int(gt_gaze_pts[0]),int(gt_gaze_pts[1])), radius=5, color=(0, 0, 255), thickness=5)
                frame = cv2.circle(frame, (int(pred_gaze_pts[0]),int(pred_gaze_pts[1])), radius=5, color=(0, 255, 0), thickness=5)
                # correct = torch.logical_and((torch.abs(pred_gaze_pts[0] - gt_gaze_pts[0]) <= 100.0), (torch.abs(pred_gaze_pts[1]-gt_gaze_pts[1]) <= 100.0)).sum().item()
                # print(pred_gaze_pts, gt_gaze_pts, correct)
            except Exception as e:
                print(e)
            cv2.imshow('image', frame)
            # out.write(frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            cv2.waitKey(0)
            ret, frame = capture.read()
