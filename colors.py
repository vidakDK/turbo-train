import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import KMeans


class Colors:
    def __init__(self, img_path, c):
        self.img = cv2.imread(img_path)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img_luv = cv2.cvtColor(self.img, cv2.COLOR_BGR2Luv)
        self.c = c
        self.img_path = img_path
        if c == "rgb":
            self.img_converted = self.img_rgb
        else:
            self.img_converted = self.img_luv

    def _find_clusters(self):
        image = self.img_converted.reshape((self.img_converted.shape[0] * self.img_converted.shape[1], 3))

        n_clusters = 6
        self.clusters = KMeans(n_clusters=n_clusters)
        self.clusters.fit(image)

    def _centroid_histogram(self):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        n_labels = np.arange(0, len(np.unique(self.clusters.labels_)) + 1)
        self.hist, _ = np.histogram(self.clusters.labels_, bins=n_labels)

        # normalize the histogram, such that it sums to one
        self.hist = self.hist.astype("float")
        self.hist /= self.hist.sum()

    def _plot_colors(self):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        start_x = 0

        # loop over the percentage of each cluster and the color of
        # each cluster
        for percent, color in zip(self.hist, self.clusters.cluster_centers_):
            # plot the relative percentage of each cluster
            end_x = start_x + (percent * 300)
            cv2.rectangle(
                bar, (int(start_x), 0), (int(end_x), 50),
                color.astype("uint8").tolist(), -1
            )
            start_x = end_x

        # return the bar chart
        return bar

    def find_colors(self):
        self._find_clusters()
        self._centroid_histogram()
        bar = self._plot_colors()

        # Plot color bar:
        if self.c == "luv":
            bar = cv2.cvtColor(bar, cv2.COLOR_LUV2RGB)

        return bar


if __name__ == "__main__":
    i1 = "images/9small.jpg"
    i2 = "images/15small.jpg"
    i3 = "images/small.jpg"
    i4 = "images/styled1.jpg"

    # Calculate RGB/LUV color bars
    c = Colors(i4, "luv")
    bar_luv = c.find_colors()
    plt.figure(), plt.gca().set_title('LUV'), plt.imshow(bar_luv), plt.axis("off")
    c = Colors(i4, "rgb")
    bar_rgb = c.find_colors()
    plt.figure(), plt.gca().set_title('RGB'), plt.imshow(bar_rgb), plt.axis("off")

    # Plot bars and original image:
    gs = gridspec.GridSpec(2, 2)
    plt.figure()
    plt.subplot(gs[:, 0]), plt.gca().set_title(c.img_path), plt.imshow(c.img_rgb), plt.axis("off")
    plt.subplot(gs[0, 1]), plt.gca().set_title('RGB'), plt.imshow(bar_rgb), plt.axis("off")
    plt.subplot(gs[1, 1]), plt.gca().set_title('LUV'), plt.imshow(bar_luv), plt.axis("off")
    plt.show()
