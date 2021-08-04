import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import rescale, resize, AffineTransform, warp, rotate
from PIL import Image

if __name__ == "__main__":
    folder = '/home/sanketthakur/Documents/gaze_pred/IMU-data_processing/test_BookShelf_S1/images/'
    file = 'frames_1.jpg'
    file2 = 'frames_81.jpg'
    os.chdir(folder)
    cfile = file
    transform = AffineTransform(translation=(-100, 0))
    img = cv2.imread(file)
    cimg = cv2.imread(cfile)
    cimg2 = cv2.imread(file2)
    # cimg  = warp(cimg, transform)
    # cimg = rescale(cimg, 0.25, anti_aliasing=False)
    final = np.concatenate((cimg, cimg2), axis=2)

    # final[:,:,0:3] = cv2.GaussianBlur(final[:,:,0:3], (10,10), 0)
    final  = warp(final, transform)
    final = rotate(final, angle=20)

    old = final[:,:, 0:3]
    new = final[:, :, 3:]
    # new = rotate(new, angle=20)
    # new = rescale(new, 0.25, anti_aliasing=True)
    # new = cv2.GaussianBlur(new, (10,10), 0)

    print(new.shape)
    plt.imshow(new)
    plt.show()
