import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# matplotlib.use("WX")
import matplotlib.pyplot as plt

img = cv2.imread('moon_toy.jpg', 0) #grayscale
template = cv2.imread('template_3.jpg', 0)
w, h = template.shape[::-1]

threshold = .9
# Apply template matching
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
# print res
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

loc = np.where(res >= threshold)
print len(loc[0])
for pt in zip(*loc[::-1]):
    print pt, res[pt]
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 2)

plt.imshow(img, cmap = 'gray')
plt.title('Detected Bounding Boxes'), plt.xticks([]), plt.yticks([])
plt.show()
# cv2.imshow('image', img)
# cv2.waitKey()
# k = cv2.waitKey(5) & 0xFF
# cv2.destroyAllWindows()

# plt.subplot(121), plt.imshow(res, cmap = 'gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img, cmap = 'gray')
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.suptitle(meth)
#
# plt.show()