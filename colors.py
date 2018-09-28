import cv2
import numpy as np
from matplotlib import pyplot as plt


class Colors:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.luv = luv = cv2.cvtColor(self.img, cv2.COLOR_BGR2Luv)
