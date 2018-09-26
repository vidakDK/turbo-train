import cv2
import numpy as np

img = cv2.imread("9.jpg")
luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)




# get only blue:
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
res = cv2.bitwise_and(img,img, mask= mask)
cv2.imshow('img', cv2.resize(img, (960, 540)))
cv2.imshow('mask', cv2.resize(mask, (960, 540)))
cv2.imshow('res', cv2.resize(res, (960, 540)))

# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# imS = cv2.resize(hsv, (960, 540))                    # Resize image
# cv2.imshow("output", imS)                            # Show image
cv2.waitKey(0)
cv2.destroyAllWindows()
