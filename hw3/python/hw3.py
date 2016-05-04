#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imread
from scipy.misc import imsave


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def gaussianHist(x, mu, sig):
    """Generate gaussian histogram."""
    hist = np.exp(-((x-mu)**2. / (2*(sig**2.))))

    return hist / hist.sum()


def plotHist(title, bins, histogram, fig, pos):
    """Create bar plot with given histogram data."""
    ax = fig.add_subplot(pos)
    ax.tick_params(right="off", top="off")  # Remove right and top ticks
    plt.title(title, size=15)
    plt.xlabel("Brightness value", size=15)
    plt.ylabel("Number of pixels", size=15)
    ax.set_xticks(bins)
    ax.bar(bins, histogram, align="center")


def searchNearest(array, n):
    """Resutn index of nearest element to given value in an array."""
    idx = (np.abs(array - n)).argmin()
    return idx


def histMatch(bins, histIn, mu, std, show=False):
    """Perform histogram matching with Gaussian contrast method."""
    # Get reference gaussian histogram
    histRef = gaussianHist(bins, mu, std) * (histIn.sum())

    # Compute cumulative sum
    cumIn = np.cumsum(histIn)
    cumRef = np.cumsum(histRef)

    # Perform histogram matching
    newVal = np.array(map(lambda e: searchNearest(cumRef, e), cumIn))
    histOut = map(lambda v: histIn[newVal == v].sum(), bins)
    cumOut = np.cumsum(histOut)

    if show:
        # Initialize figures
        fig = plt.figure("Histogram Equalization", figsize=(18, 8))

        # For input histogram
        plotHist("Input histogram", bins, histIn, fig, 231)

        # For input histogram
        plotHist("Cumulative histogram of input data", bins, cumIn, fig, 234)

        # For reference histogram
        plotHist("Reference histogram", bins, histRef, fig, 232)

        # For cumulative sum of reference histogram
        plotHist(
            "Cumulative histogram of reference data", bins, cumRef, fig, 235)

        # For output histogram
        plotHist("Output histogram", bins, histOut, fig, 233)

        # For cumulative sum of output histogram
        plotHist("Cumulative histogram of output data", bins, cumOut, fig, 236)

        plt.tight_layout()
        plt.show()

    return newVal


def gaussianEnhance(inputFileName, outputFileName, show=False):
    """Perform image enhancement with Gaussian contrast method."""
    # Read image
    inputImg = imread(inputFileName, "L")
    mu = 128
    std = inputImg.std()

    # Generate histogram
    histIn, _ = np.histogram(inputImg, bins=256)

    # Histogram matching
    newVal = histMatch(np.arange(256), histIn, mu, std, show)
    outputImg = np.zeros(inputImg.shape)
    for v in np.arange(256):
        outputImg[inputImg == v] = newVal[v]

    # Output result
    imsave(outputFileName, outputImg.astype(np.uint8))


def edgeDection(inputFileName, threshold=(100, 200)):
    """Perform canny edge dection."""
    # Read image
    img = cv2.imread(inputFileName, 0)

    # Detect line feature
    edgeImg = cv2.Canny(img, *threshold)

    # Save result
    fileName, ext = os.path.splitext(inputFileName)
    imsave("".join([fileName, "_edge", ext]), edgeImg)


def main():
    # --- For question1 ---
    # Define input histogram
    bins = np.arange(17)
    histIn = np.array([0, 0, 7, 4, 4, 2, 2, 0, 0, 0, 0, 2, 4, 3, 2, 6, 0])

    # Define parameters for gaussian function
    mu = 8      # Mean
    std = 2     # Standard deviation
    histMatch(bins, histIn, mu, std, show=True)

    # --- For question2 ---
    inputFileName = "5941.tif"
    outputFileName = "5941_enhanced.tif"
    gaussianEnhance(inputFileName, outputFileName, show=False)

    # Canny edge detection
    edgeDection(inputFileName, (100, 200))
    edgeDection(outputFileName, (100, 200))


if __name__ == '__main__':
    main()
