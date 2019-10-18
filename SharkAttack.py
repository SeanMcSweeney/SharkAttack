import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui

# Shark Attack !!!
# Assignment 1

# The first step is to convert to a different colourspace
# After trial and error it looked like converting to the HSV colourspace was the most effective

# the second part is to enhance the images to increase contrast
# Equalising the image increases its contrast and makes the image sharper

# the third part is to extract the fish from the image and convert the background to white
# the fourth part is to further enhance the fish
# the final part is to crop and rotate the image to contain only the shark (optional)

#step one
f = easygui.fileopenbox()
I = cv2.imread(f)
YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
Edit = HSV[:,:,2]
new = YUV[:,:,1]
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

#step two
ContrastUp = cv2.equalizeHist(Edit)

#Kernel
k = np.array([[-1,2,-1], [2,8,2], [-1,2,-1]], dtype=float)
k = k / 12

withFilter = cv2.filter2D(I, ddepth=-1, kernel=k)


#step three
#Thresholding

T = 50
T, Simp = cv2.threshold(ContrastUp, thresh = T, maxval = 255, type = cv2.THRESH_BINARY_INV)

Adapt = cv2.adaptiveThreshold(ContrastUp, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 7, C = 8)

#More Masks
shape = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
Boundary = cv2.morphologyEx(Simp,cv2.MORPH_GRADIENT,shape)
NewMask = cv2.dilate(Simp,shape)

# Using Masks
ROI1 = cv2.bitwise_and(I,I,mask=Adapt)
ROI2 = cv2.bitwise_and(I,I,mask=Simp)
ROI = cv2.bitwise_and(ROI1,ROI2)

#cv2.imshow("image", Edit)
#cv2.imshow("image", G)
#cv2.imshow("imageC", ContrastUp)
cv2.imshow("image", Simp)
#cv2.imshow("image", ROI)
#cv2.imshow("image", withFilter)
#cv2.imshow("image", Adapt)
#cv2.imshow("image", NewMask)
#cv2.imshow("image", Boundary)
key = cv2.waitKey(0)
cv2.destroyAllWindows()