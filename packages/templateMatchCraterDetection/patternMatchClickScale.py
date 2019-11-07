import cv2
import numpy as np
import imutils
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use("WX")
import matplotlib.pyplot as plt

cropping = False

x_start, y_start, x_end, y_end = 10, 10, 40, 40

image = cv2.imread('moon_toy.jpg', 0) #'moon_toy.jpg'
# print image.shape
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

def showBoulderLocations(image, template, threshold = 0.6):
    print "hello"
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
        # for i in range(res.shape[0]):
        #     for j in range(res.shape[1]):
        #         loc = (i, j)
        #         corr_coeff = res[loc]
        #         if corr_coeff > found[int(i * r), int(j * r), 0]:
        #             found[int(i * r), int(j * r)] = (corr_coeff, r)
    loc = np.where(found[:, :, 0] >= threshold)
    for pt in zip(*loc):
        (i, j) = pt
        r = found[i, j, 1]
        (startX, startY) = (j, i)
        (endX, endY) = (int(j + r * h), int(i + r * w))
        cv2.rectangle(image, (startX, startY), (endX, endY), 255, 2)
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         corr_coeff = found[i, j, 0]
    #         r = found[i, j, 1]
    #         if corr_coeff > threshold:
    #             (startX, startY) = (j, i)
    #             (endX, endY) = (int(j + r * w), int(i + r * h))
    #             cv2.rectangle(image, (startX, startY), (endX, endY), 255, 2)
    cv2.imshow("image", image)

while True:

    i = image.copy()

    if not cropping:
        print "not cropping"
        # cv2.imshow("image", image)
        # template = i[x_start: x_end, y_start: y_end]
        template = i[y_start: y_end, x_start: x_end]
        showBoulderLocations(i, template)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), 255, 2)
        cv2.imshow("image", i)

    cv2.waitKey(2)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# close all open windows
cv2.destroyAllWindows()