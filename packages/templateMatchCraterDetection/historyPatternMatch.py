import cv2
import numpy as np
import imutils
from nms import non_max_suppression_fast
import matplotlib
matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import time

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0
history_dictionary = {}
all_picks = None

image = cv2.imread('moon_toy.jpg', 0) #'LROC_craters_cropped.png'
image_umat = cv2.UMat(image)
# assertIsInstance(image, cv2.UMat)
# print image.shape
oriImage = image.copy()

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, image, image_umat, oriImage, cropping, history_dictionary

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is be
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False  # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if x_start != x_end or y_start != y_end:  # when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            # plt.imshow(roi, cmap="gray")
            cv2.imshow("cropped", cv2.UMat(roi))
            cv2.resizeWindow("cropped", (300,300))

            template = oriImage[y_start: y_end, x_start: x_end]
            showCraterLocations(image, image_umat, template)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", (600,600))
cv2.setMouseCallback("image", mouse_crop)

def showCraterLocations(image, image_umat, template, threshold = 0.6):
    global all_picks
    w, h = template.shape
    found = np.zeros((image.shape[0], image.shape[1]), dtype=(float, 2))
    for scale in np.linspace(0.5, 2.5, 10):
        resized = imutils.resize(image, width=int(image.shape[1] * scale))
        r = 1.0 / scale
        if resized.shape[1] < h or resized.shape[0] < w:
            break
        res = cv2.matchTemplate(cv2.UMat(resized), cv2.UMat(template), cv2.TM_CCOEFF_NORMED).get()
        loc = np.where(res >= threshold)
        for pt in zip(*loc):
            (i, j) = pt
            corr_coeff = res[i, j]
            if corr_coeff > found[int(i * r), int(j * r)][0]:
                found[int(i * r), int(j * r)] = (corr_coeff, r)
    loc = np.where(found[:, :, 0] >= threshold)
    boxes = []
    for pt in zip(*loc):
        (i, j) = pt
        r = found[i, j, 1]
        (startX, startY, endX, endY) = (j, i, int(j + r * h), int(i + r * w))
        boxes.append((startX, startY, endX, endY))

    # perform non-maximum suppression on the bounding boxes
    print "Initial length:", len(boxes)
    boxes = np.array(boxes)
    if all_picks is not None and len(boxes) > 0:
        boxes = np.concatenate((boxes, all_picks))
    pick = non_max_suppression_fast(boxes)
    print pick
    print "After applying non-maximum suppression:", len(pick)
    if all_picks is None:
        all_picks = np.array(pick)
    elif len(pick) > 0:
        all_picks = np.concatenate((all_picks, np.array(pick)))
    # loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image_umat, (startX, startY), (endX, endY), 255, 2)

    cv2.imshow("image", image_umat)
    # plt.imshow(image, cmap="gray")

firstLoop = True
while True:
    if firstLoop:
        cv2.imshow("image", image_umat)
        firstLoop = False
        #press space bar to start interactive session
        cv2.waitKey(0)
    if cropping:
        cv2.rectangle(image_umat, (x_start, y_start), (x_end, y_end), 255, 2)
        # plt.imshow("image", image)
        cv2.imshow("image", image_umat)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(1)

# close all open windows
cv2.destroyAllWindows()