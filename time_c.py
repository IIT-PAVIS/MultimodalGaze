#import numpy as np 
#from torchvision import transforms 
#from torch import from_numpy
#import torch 
from torch import load 

file = '/home/sanketthakur/Documents/gaze_pred/IMU-data_processing/training_images/frames_0.pt'
a = load(file)
print(a.shape)
#a = from_numpy(np.load(file))

#tf = transforms.Compose([transforms.ToTensor()])

#d = tf(a)




