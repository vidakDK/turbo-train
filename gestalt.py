import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from scipy.optimize import brute, minimize_scalar

# img = cv2.imread("9.jpg")



class ColorExtractor:

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.luv = luv = cv2.cvtColor(self.img, cv2.COLOR_BGR2Luv)
        print(luv.min, luv.max)
        self.hist_lightness = cv2.calcHist([luv], [0], None, [256], [0, 256])
        self.hist_u = cv2.calcHist([luv], [1], None, [256], [0, 256])
        self.hist_v = cv2.calcHist([luv], [2], None, [256], [0, 256])

        plt.figure(1)
        plt.plot(self.hist_lightness, label='lightness')
        plt.legend()

        plt.show()

    def k(self, t):
        """
        Function that calculates the clustering metric k, based on an input threshold. Small k means
        that the threshold is good.
        :param t: Threshold for which metric k is calculated.
        :return: the value of the k metric.
        """
        d0 = np.std(self.hist_lightness)**2
        n0 = np.sum(self.hist_lightness)

        # Create masks given threshold t, and calculate histograms:
        lower_mask = cv2.inRange(self.luv, np.array([0, 0, 0]), np.array([t - 1, 255, 255]))
        higher_mask = cv2.inRange(self.luv, np.array([t, 0, 0]), np.array([255, 255, 255]))
        lower_hist = cv2.calcHist([self.luv], [0], lower_mask, [256], [0, 256])
        higher_hist = cv2.calcHist([self.luv], [0], higher_mask, [256], [0, 256])

        # Calculate Dispersion and number of pixels:
        d1 = np.std(lower_hist)
        n1 = np.sum(lower_hist)
        d2 = np.std(higher_hist)
        n2 = np.sum(higher_hist)

        # Calculate metric k:
        return (sqrt(d1*n1) + sqrt(d2*n2)) / (sqrt(d0*n0))

    def min_k(self):
        lst = []
        for t in range(1, 255):
            lst.append((t, self.k(t)))
        ts, ks = zip(*lst)
        plt.figure(4)
        plt.plot(ks, label='approach 1')
        plt.legend()
        # plt.show()
        return min(lst, key=lambda x: x[1])

    def threshold_image(self, t):
        lower_mask = cv2.inRange(self.luv, np.array([0, 0, 0]), np.array([t - 1, 255, 255]))
        higher_mask = cv2.inRange(self.luv, np.array([t, 0, 0]), np.array([255, 255, 255]))
        lower_masked_image = cv2.bitwise_and(self.img, self.img, mask=lower_mask)
        higher_masked_image = cv2.bitwise_and(self.img, self.img, mask=higher_mask)
        return lower_masked_image, higher_masked_image, lower_mask, higher_mask

    def do_cool_stuff(self):
        t = 64
        lower_image, higher_image, lower_mask, higher_mask = self.threshold_image(t)
        lower_hist = cv2.calcHist([self.luv], [0], lower_mask, [256], [0, 256])
        higher_hist = cv2.calcHist([self.luv], [0], higher_mask, [256], [0, 256])
        plt.figure(2)
        plt.subplot(321), plt.imshow(self.img)
        plt.subplot(322), plt.plot(self.hist_lightness)
        plt.subplot(323), plt.imshow(lower_image)
        plt.subplot(324), plt.plot(lower_hist)
        plt.subplot(325), plt.imshow(higher_image)
        plt.subplot(326), plt.plot(higher_hist)


# Get brightness histogram
# hist = cv2.calcHist([luv], [0], None, [256], [0, 256])
# hist,bins = np.histogram(img.ravel(),256,[0,256])

# plot images
# cv2.imshow('first_cluster', cv2.resize(first_cluster, (960, 540)))
#         cv2.imshow('second_cluster', cv2.resize(second_cluster, (960, 540)))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


if __name__ == "__main__":
    c = ColorExtractor("9.jpg")
    # print(c.k(64), c.k(128), c.k(192))
    # print(c.k(64))
    # plt.legend()
    # plt.show()
    # t, k = c.min_k()
    # print(t,k)
    # c.threshold_image(t)
# # get only blue:
# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])
# lower_yellow = np.array([20, 100, 100])
# upper_yellow = np.array([30, 255, 255])
# # mask = cv2.inRange(hsv, lower_blue, upper_blue)
# mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# res = cv2.bitwise_and(img,img, mask= mask)
# cv2.imshow('img', cv2.resize(img, (960, 540)))
# cv2.imshow('mask', cv2.resize(mask, (960, 540)))
# cv2.imshow('res', cv2.resize(res, (960, 540)))
#
# # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# # imS = cv2.resize(hsv, (960, 540))                    # Resize image
# # cv2.imshow("output", imS)                            # Show image
# cv2.waitKey(0)
# cv2.destroyAllWindows()
