import os, cv2
from tqdm import tqdm
import torch, argparse
from pathlib import Path
import numpy as np
from helpers import Helpers
import torch.nn as nn
from vision_pipeline import VISION_PIPELINE, VIS_FINAL_DATASET
from signal_pipeline import IMU_PIPELINE, SIG_FINAL_DATASET
from pipeline_new import FusionPipeline, FINAL_DATASET
from variables import RootVariables

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    var = RootVariables()
    # test_folder = 'test_InTheDeak_S2'
    lastFolder, newFolder = None, None
    for index, subDir in enumerate(sorted(os.listdir(var.root))):
        #if 'train_BookShelf' in subDir:
        #    continue
        if 'train_' in subDir:
            newFolder = subDir
            os.chdir(var.root)

            test_folder = 'test_' + newFolder[6:]
            _ = os.system('mv ' + newFolder + ' test_' + newFolder[6:])
            if lastFolder is not None:
                print('Last folder changed')
                _ = os.system('mv test_' + lastFolder[6:] + ' ' + lastFolder)

            print(newFolder, lastFolder)

            # test_folder = 'test_BookShelf_S1'
            vision_model_checkpoint = 'vision_checkpointAdam9CNN_' + test_folder[5:] + '.pth'
            signal_model_checkpoint = 'signal_checkpoint0_' + test_folder[5:] + '.pth'
            flownet_checkpoint = 'flownets_EPE1.951.pth.tar'

            trim_frame_size = 150
            pipeline = VISION_PIPELINE(flownet_checkpoint)
            pipeline = IMU_PIPELINE()
            print(pipeline)
            criterion = nn.L1Loss()
            # if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + vision_model_checkpoint).is_file():
            #     checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + vision_model_checkpoint,  map_location="cuda:0")
            #     pipeline.load_state_dict(checkpoint['model_state_dict'])
            #     # pipeline.current_loss = checkpoint['loss']
            #     print('Model loaded')

            if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + signal_model_checkpoint).is_file():
                checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + signal_model_checkpoint,  map_location="cuda:0")
                pipeline.load_state_dict(checkpoint['model_state_dict'])
                # pipeline.current_loss = checkpoint['loss']
                print('Model loaded')


            # if 'test_Book' in test_folder:
            #     utils = Helpers(test_folder)
            # else:
            utils = Helpers(test_folder, reset_dataset=1)
            _, imu_testing_feat, _, testing_target = utils.load_datasets()
            os.chdir(pipeline.var.root)

        #     pipeline.eval()
        #     with torch.no_grad():
        #         testDataset = SIG_FINAL_DATASET(imu_testing_feat, testing_target)
        #
        #         testLoader = torch.utils.data.DataLoader(testDataset, shuffle=False, batch_size=pipeline.var.batch_size, drop_last=True, num_workers=0)
        #
        # #        tqdm_trainLoader = tqdm(trainLoader)
        #         tqdm_testLoader = tqdm(testLoader)
        #
        #         num_samples = 0
        #         total_loss, total_correct, total_accuracy = [], 0.0, 0.0
        #         predList, labelList, testPD = None, None, None
        #         for batch_index, (feat, labels) in enumerate(tqdm_testLoader):
        #             num_samples += feat.size(0)
        #             labels = labels[:,0,:]
        #             # labels[:,0] *= 0.2667
        #             # labels[:,1] *= 0.3556
        #             pred = pipeline(feat.float()).to("cuda:0")
        #
        #             loss = criterion(pred, labels.float())
        #             pred, labels = pipeline.get_original_coordinates(pred, labels)
        #
        #             dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
        #             if batch_index > 0:
        #                 testPD = torch.cat((testPD, dist), 1)
        #                 predList = torch.cat((predList, pred), 0)
        #                 labelList = torch.cat((labelList, labels), 0)
        #             else:
        #                 testPD = dist
        #                 predList = pred
        #                 labelList = labels
        #
        #             total_loss.append(loss.detach().item())
        #             total_correct += pipeline.get_num_correct(pred, labels.float())
        #             total_accuracy = total_correct / num_samples
        #             tqdm_testLoader.set_description('training: ' + '_loss: {:.4} correct: {} accuracy: {:.3} Mean dist: {} STD dist: {}'.format(
        #                 np.mean(total_loss), total_correct, 100.0*total_accuracy, torch.mean(testPD), torch.std(testPD)))

            pipeline = FusionPipeline(flownet_checkpoint, test_folder)
            # print(pipeline)

            model_checkpoint = 'pipeline_checkpointAdam_' + test_folder[5:] + '.pth'
            if Path(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint).is_file():
                checkpoint = torch.load(pipeline.var.root + 'datasets/' + test_folder[5:] + '/' + model_checkpoint,  map_location="cuda:0")
                pipeline.load_state_dict(checkpoint['model_state_dict'])
                print('Model loaded')

            pipeline.eval()
            testPD, dummyPD = None, None

            with torch.no_grad():
                testDataset = FINAL_DATASET('testing_images', imu_testing_feat, testing_target)
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
                    loss = criterion(pred.float(), labels.float())
                    pred, labels = pipeline.get_original_coordinates(pred, labels)

                    dist = torch.cdist(pred, labels.float(), p=2)[0].unsqueeze(dim=0)
                    dummDist = torch.cdist(dummy_pts, labels.float(), p=2)[0].unsqueeze(dim=0)
                    if batch_index > 0:
                        testPD = torch.cat((testPD, dist), 1)
                        dummyPD = torch.cat((dummyPD, dummDist), 1)
                    else:
                        testPD = dist
                        dummyPD = dummDist

                    total_loss.append(loss.detach().item())
                    total_correct += pipeline.get_num_correct(pred, labels.float())
                    dummy_correct += pipeline.get_num_correct(dummy_pts.float(), labels.float())
                    dummy_accuracy = dummy_correct / num_samples
                    total_accuracy = total_correct / num_samples
                    tqdm_testLoader.set_description('testing: ' + '_loss: {:.4} correct: {} accuracy: {:.3} DAcc: {:.4} Mean dist: {} SD dist: {} dummMean : {} dummSTD: {}'.format(
                        np.mean(total_loss), total_correct, 100.0*total_accuracy,  np.floor(100.0*dummy_accuracy), torch.mean(testPD), torch.std(testPD), torch.mean(dummyPD), torch.std(dummyPD)))

            lastFolder = newFolder
