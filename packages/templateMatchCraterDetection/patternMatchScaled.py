import cv2
import numpy as np
import imutils
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use("WX")
import matplotlib.pyplot as plt

img = cv2.imread('moon_toy.jpg', 0)
template = cv2.imread('small_template.jpg', 0)
w, h = template.shape[::-1]
found = np.zeros((img.shape[0], img.shape[1]), dtype=(float, 2))
threshold = 0.6
for scale in np.linspace(0.5, 2.5, 10):
    # resized = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)), interpolation=cv2.INTER_AREA)
    resized = imutils.resize(img, width=int(img.shape[1] * scale))
    r = 1.0/scale
    # r = img.shape[1]/float(resized.shape[1])
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            loc = (i, j)
            corr_coeff = res[loc]
            if corr_coeff > found[int(i*r), int(j*r)][0]:
                found[int(i*r), int(j*r)] = (corr_coeff, r)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        corr_coeff = found[i][j][0]
        r = found[i][j][1]
        if corr_coeff > threshold:
            (startX, startY) = (j, i)
            (endX, endY) = (int(j+r*w), int(i+r*h))
            # (startX, startY) = (int(i * r), int(j * r))
            # (endX, endY) = (int((i + w) * r), int((j + h) * r))
            # print (startX, startY), (endX, endY)
            cv2.rectangle(img, (startX, startY), (endX, endY), 255, 2)
plt.imshow(img, cmap = 'gray')
plt.title('Detected Bounding Boxes'), plt.xticks([]), plt.yticks([])
plt.show()
# cv2.imshow("Image", img)
# cv2.waitKey(0)

