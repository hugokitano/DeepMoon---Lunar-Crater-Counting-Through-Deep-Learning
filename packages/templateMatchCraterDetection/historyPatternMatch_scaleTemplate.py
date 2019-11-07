import cv2
import numpy as np
import imutils
from nms import non_max_suppression_fast
import matplotlib
matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import time
from collections import defaultdict

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0
history_dictionary = defaultdict(set)
old_bounding_boxes = None

image = cv2.imread('LROC_craters.jpg', 0) #'LROC_craters_cropped.png'
image_umat = cv2.UMat(image)
# assertIsInstance(image, cv2.UMat)
# print image.shape
oriImage = image.copy()

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, image, image_umat, oriImage, cropping

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

#called each time user selects a new crater
def showCraterLocations(image, image_umat, template, threshold = 0.6):
    global x_start, y_start, x_end, y_end, old_bounding_boxes, history_dictionary
    w, h = image.shape
    w_template, h_template = template.shape
    #locations are top-left corner of corresponding bounding box
    res_across_scales = np.zeros((image.shape[0], image.shape[1]), dtype=(float, 2))
    for scale in np.linspace(0.5, 1.5, 10):
        resized_template = imutils.resize(template, width=int(template.shape[1] * scale),
                                          height = int(template.shape[0] * scale))
        if resized_template.shape[0] > w or resized_template.shape[1] > h:
            break
        #change image and resized_template to UMat?
        res = cv2.matchTemplate(image_umat, cv2.UMat(resized_template), cv2.TM_CCOEFF_NORMED).get()
        loc = np.where(res >= threshold)
        for pt in zip(*loc):
            (i, j) = pt
            corr_coeff = res[i, j]
            if corr_coeff > res_across_scales[i, j][0]:
                res_across_scales[i, j] = (corr_coeff, scale)
    strong_match_locs = np.where(res_across_scales[:, :, 0] >= threshold)
    new_bounding_boxes = []
    for pt in zip(*strong_match_locs):
        (i, j) = pt
        best_scale = res_across_scales[i, j, 1]
        (startX, startY, endX, endY) = (j, i, int(j + best_scale * h_template), int(i + best_scale * w_template))
        new_bounding_boxes.append((startX, startY, endX, endY))

    # perform non-maximum suppression on the bounding boxes
    print "Initial length:", len(new_bounding_boxes)
    new_bounding_boxes = np.array(new_bounding_boxes)
    new_pick = list(non_max_suppression_fast(new_bounding_boxes))
    new_pick = [list(arr) for arr in new_pick]

    if old_bounding_boxes is not None:
        old_pick = list(non_max_suppression_fast(old_bounding_boxes))
        old_pick = [list(arr) for arr in old_pick]

    if old_bounding_boxes is not None and len(new_bounding_boxes) > 0:
        new_bounding_boxes = np.concatenate((new_bounding_boxes, old_bounding_boxes))
    both_pick = list(non_max_suppression_fast(new_bounding_boxes))
    both_pick = [list(arr) for arr in both_pick]

    new_pick = [ls for ls in new_pick if ls in both_pick]
    if old_bounding_boxes is not None:
        intersection = [ls for ls in both_pick if ls in old_pick]
        new_pick = [ls for ls in new_pick if ls not in intersection]

    history_dictionary[(x_start, y_start, x_end, y_end)] = new_pick
    print "After applying non-maximum suppression:", len(new_pick)

    # loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in new_pick:
        cv2.rectangle(image_umat, (startX, startY), (endX, endY), 255, 2)
    if old_bounding_boxes is None:
        old_bounding_boxes = np.array(new_pick)
    elif len(new_pick) > 0:
        print old_bounding_boxes.shape, np.array(new_pick).shape
        old_bounding_boxes = np.concatenate((old_bounding_boxes, np.array(new_pick)))

    cv2.imshow("image", image_umat)

firstLoop = True
while True:
    if firstLoop:
        cv2.imshow("image", image_umat)
        firstLoop = False
        #press space bar to start interactive session
        cv2.waitKey(0)
    if cropping:
        cv2.rectangle(image_umat, (x_start, y_start), (x_end, y_end), 255, 2)
        cv2.imshow("image", image_umat)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(1)

cv2.imwrite('craters_with_bounding_boxes.png',image)
# close all open windows
cv2.destroyAllWindows()