import cv2
import numpy as np
import imutils
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use("WX")
import matplotlib.pyplot as plt

cropping = False

x_start, y_start, x_end, y_end = 10, 10, 40, 40

image = cv2.imread('LROC_craters.jpg', 0) #'butterfly.jpeg'
print image.shape
oriImage = image.copy()

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
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

def showBoulderLocations(image, template, threshold = 0.75):
    w, h = template.shape
    # Apply template matching
    res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED) #TM_CCOEFF_NORMED
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + h, pt[1] + w), 255, 2)

    cv2.imshow("image", image)

while True:

    i = image.copy()

    if not cropping:
        # cv2.imshow("image", image)
        # template = i[x_start: x_end, y_start: y_end]
        template = i[y_start: y_end, x_start: x_end]
        showBoulderLocations(i, template)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), 255, 2)
        cv2.imshow("image", i)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close all open windows
cv2.destroyAllWindows()