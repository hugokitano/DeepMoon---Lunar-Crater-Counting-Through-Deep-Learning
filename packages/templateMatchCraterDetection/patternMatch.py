import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use("WX")
import matplotlib.pyplot as plt

img = cv2.imread('moon_toy.jpg', 0)
img2 = img.copy()
template = cv2.imread('template_2.jpg', 0)
w, h = template.shape[::-1]

# threshold = 0.8
# ret = True
# while ret:
#     if len(res):
#         loc = np.where( res >= threshold)
#         for pt in zip(*loc[::-1]):
#             cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#         # cv2.imwrite('res.png',img_rgb)
#         cv2.imshow('image', img)
#         k = cv2.waitKey(5) & 0xFF
#         if k == 27:
#             break
# cv2.destroyAllWindows()

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    # img = np.asarray(img)
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()