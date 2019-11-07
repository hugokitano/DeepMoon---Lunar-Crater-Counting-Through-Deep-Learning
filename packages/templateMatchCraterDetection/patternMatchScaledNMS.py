import cv2
import numpy as np
import imutils
from nms import non_max_suppression_fast
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use("WX")
import matplotlib.pyplot as plt

img = cv2.imread('moon_toy.jpg', 0)
orig = img.copy()
template = cv2.imread('small_template.jpg', 0)
w, h = template.shape[::-1]
found = np.zeros((img.shape[0], img.shape[1]), dtype=(float, 2))
threshold = 0.6
for scale in np.linspace(0.5, 2.5, 10):
    resized = imutils.resize(img, width=int(img.shape[1] * scale))
    r = 1.0/scale
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            loc = (i, j)
            corr_coeff = res[loc]
            if corr_coeff > found[int(i*r), int(j*r)][0]:
                found[int(i*r), int(j*r)] = (corr_coeff, r)
boxes = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        corr_coeff = found[i][j][0]
        r = found[i][j][1]
        if corr_coeff > threshold:
            (startX, startY, endX, endY) = (j, i, int(j+r*w), int(i+r*h))
            boxes.append((startX, startY, endX, endY))

# loop over the bounding boxes for each image and draw them
# for (startX, startY, endX, endY) in boxes:
#     cv2.rectangle(img, (startX, startY), (endX, endY), 255, 2)

# perform non-maximum suppression on the bounding boxes
print "Initial length:", len(boxes)
pick = non_max_suppression_fast(np.array(boxes))
print "After applying non-maximum suppression:", len(pick)

# loop over the picked bounding boxes and draw them
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(img, (startX, startY), (endX, endY), 255, 2)

# display the images
cv2.imshow("Original", orig)
cv2.imshow("After NMS", img)
cv2.waitKey(0)