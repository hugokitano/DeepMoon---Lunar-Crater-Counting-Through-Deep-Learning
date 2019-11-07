import cv2

print(cv2.ocl.haveOpenCL())
cv2.ocl.setUseOpenCL(True)
print(cv2.ocl.useOpenCL())