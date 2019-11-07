import cv2
import numpy as np
import imutils
from nms import non_max_suppression_fast
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use("WX")
import matplotlib.pyplot as plt

cropping = False

x_start, y_start, x_end, y_end = 10, 10, 40, 40

image = cv2.imread('moon_toy.jpg', 0)
# print image.shape
oriImage = image.copy()

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

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

        if len(refPoint) == 2:  # when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

def showCraterLocations(image, template, threshold = 0.6):
    w, h = template.shape
    found = np.zeros((image.shape[0], image.shape[1]), dtype=(float, 2))
    for scale in np.linspace(0.5, 2.5, 10):
        # resized = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)), interpolation=cv2.INTER_AREA)
        resized = imutils.resize(image, width=int(image.shape[1] * scale))
        r = 1.0 / scale
        # r = img.shape[1]/float(resized.shape[1])
        if resized.shape[1] < h or resized.shape[0] < w:
            break
        res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc):
            (i, j) = pt
            corr_coeff = res[i, j]
            if corr_coeff > found[int(i * r), int(j * r)][0]:
                found[int(i * r), int(j * r)] = (corr_coeff, r)
    boxes = []
    loc = np.where(found[:, :, 0] >= threshold)
    for pt in zip(*loc):
        (i, j) = pt
        r = found[i, j, 1]
        (startX, startY, endX, endY) = (j, i, int(j + r * h), int(i + r * w))
        boxes.append((startX, startY, endX, endY))

    # perform non-maximum suppression on the bounding boxes
    print "Initial length:", len(boxes)
    pick = non_max_suppression_fast(np.array(boxes))
    print "After applying non-maximum suppression:", len(pick)

    # loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(image, (startX, startY), (endX, endY), 255, 2)

    cv2.imshow("image", image)

while True:

    i = image.copy()

    if not cropping:
        # cv2.imshow("image", image)
        # template = i[x_start: x_end, y_start: y_end]
        template = i[y_start: y_end, x_start: x_end]
        showCraterLocations(i, template)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), 255, 2)
        cv2.imshow("image", i)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close all open windows
cv2.destroyAllWindows()