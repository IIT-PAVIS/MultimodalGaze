import numpy as np
import torch
import torch.nn as nn
import os, cv2, sys
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
sys.path.append('../')
from resnetpytorch.models import resnet
from torchvision import transforms

def create_clips(cap, index):
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/clips/output_' + str(index) + '.avi', fourcc, 5.0, (512,384), 0)
    for i in range(5):
        _, frame = cap.read()
        frame = cv2.resize(frame, (512, 384))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        out.write(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES,150+index-4)

    out.release()

if __name__ == '__main__':
    df = pd.read_csv('/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/gaze_file.csv').to_numpy()
    file3d = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/clips/output_0.avi'
    cap = cv2.VideoCapture(file3d)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    model = resnet.generate_model(50)
    transforms = transforms.Compose([transforms.ToTensor()])
    print(model)
    last = None
    _, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    last = frame
    last = transforms(last)
    last = last.unsqueeze(dim=3)
    print(last.shape)
    for i in range(4):
        print(i)
        _, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # frame = np.expand_dims(frame, axis=2)
        frame = transforms(frame)
        frame = frame.unsqueeze(dim=3)
        last = torch.cat((last, frame), axis=3)
        # cv2.imshow('img', frame)
        # cv2.waitKey(0)

    # last = np.expand_dims(last, axis=0)
    # last = transforms(last)
    last = last.unsqueeze(dim=0)
    # last = last.unsqueeze(dim=4)
    print(last.shape)

    x = model(last)
    print(x.shape)

    # file = '/Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/scenevideo.mp4'
    # cap = cv2.VideoCapture(file)
    # frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # trim_size = 150
    # _ = os.system('rm -r /Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/clips/')
    # _ = os.system('mkdir /Users/sanketsans/Downloads/Pavis_Social_Interaction_Attention_dataset/test_BookShelf_S1/clips/')
    # cap.set(cv2.CAP_PROP_POS_FRAMES,150-4)
    # for i in tqdm(range(frame_count-300)):
    #     create_clips(cap, i)
    #     ret, frame = cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     pts = [0.5, 0.5]
    #     try:
    #         gpts = list(map(literal_eval, df[trim_size+i, 1:]))
    #         avg = [sum(y) / len(y) for y in zip(*gpts)]
    #
    #         start_point = (int(pts[0]*frame.shape[1]) - 100, int(pts[1]*frame.shape[0]) + 100)
    #         end_point = (int(pts[0]*frame.shape[1]) + 100, int(pts[1]*frame.shape[0]) - 100)
    #         pred_start_point = (int(avg[0]*frame.shape[1]) - 100, int(avg[1]*frame.shape[0]) + 100)
    #         pred_end_point = (int(avg[0]*frame.shape[1]) + 100, int(avg[1]*frame.shape[0]) - 100)
    #
    #         frame = cv2.circle(frame, (int(pts[0]*1920),int(pts[1]*1080)), radius=5, color=(0, 0, 255), thickness=5)
    #         frame = cv2.circle(frame, (int(avg[0]*1920),int(avg[1]*1080)), radius=5, color=(0, 255, 0), thickness=5)
    #
    #         frame = cv2.rectangle(frame, start_point, end_point, color=(0, 0, 255), thickness=5)
    #         frame = cv2.rectangle(frame, pred_start_point, pred_end_point, color=(0, 255, 0), thickness=5)
    #     except:
    #         pass
    #     cv2.imshow('image', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
