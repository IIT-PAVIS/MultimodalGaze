import cv2, os, sys
import numpy as np
sys.path.append('../')
from loader import JSON_LOADER
from variables import RootVariables

def get_num_correct(pred, label):
    return np.logical_and((np.abs(pred[0] - label[0]) <= 100.0), (np.abs(pred[1]-label[1]) <= 100.0)).sum().item()
    # return torch.logical_and((torch.abs(pred[:,0]*1920-label[:,0]*1920) <= 100.0), (torch.abs(pred[:,1]*1080-label[:,1]*1080) <= 100.0)).sum().item()

if __name__ == "__main__":
    var = RootVariables()
    folder = 'train_shahid_Lift_S1/'
    uni = None
    os.chdir(var.root + folder)
    capture = cv2.VideoCapture('scenevideo.mp4')
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    dataset = JSON_LOADER(folder)
    dataset.POP_GAZE_DATA(frame_count)
    gaze_arr = np.array(dataset.var.gaze_data).transpose()
    temp = np.zeros((frame_count*4-var.trim_frame_size*4*2 - 4, 2))
    temp[:,0] = gaze_arr[tuple([np.arange(var.trim_frame_size*4 +4, frame_count*4 - var.trim_frame_size*4), [0]])]
    temp[:,1] = gaze_arr[tuple([np.arange(var.trim_frame_size*4 +4, frame_count*4 - var.trim_frame_size*4), [1]])]

    capture = cv2.VideoCapture('scenevideo.mp4')
    capture.set(cv2.CAP_PROP_POS_FRAMES,var.trim_frame_size)
    ret, frame = capture.read()

    correct, total_pts = 0, 0
    for index in range(frame_count - 300):
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 512, 512)
        gaze = temp[4*index]
        try:
            gaze[0] = gaze[0]*frame.shape[1]
            gaze[1] = gaze[1]*frame.shape[0]
            dummy_pts = (np.ones(2) * 0.5)
            dummy_pts[0] *= 1920.0
            dummy_pts[1] *= 1080.0
            frame = cv2.circle(frame, (int(gaze[0]),int(gaze[1])), radius=5, color=(0, 0, 255), thickness=5)
            correct += get_num_correct(dummy_pts, gaze)
            total_pts += 1
        except:
            pass
        # index += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(0)
        ret, frame = capture.read()

    print('accuracy: ', 100.0*(correct / total_pts))
