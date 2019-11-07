import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import os

path = 'LROC_craters_cropped.png'
if os.path.isfile(path):
    img = cv2.imread(path, 0)
    print img
    cv2.imshow('ImageWindow', img)
    # plt.imshow(img, cmap='gray')
    cv2.waitKey()
else:
    print ("The file " + " does not exist.")
