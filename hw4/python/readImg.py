#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module is for loading binary image file."""
import cv2
import numpy as np
import struct


def readBsq(imgName, numRow, numCol, numBand, dataFormat):
    """Read an binary image with BSQ format."""
    inputImg = open(imgName, 'rb')
    bytePerPix = struct.calcsize(dataFormat)    # Get the size of data format

    outputImg = np.empty((numRow, numCol, numBand))
    for band in range(numBand):
        for row in range(numRow):
            for col in range(numCol):
                data = inputImg.read(bytePerPix)    # Read pixel binary value

                #  Unpack and store the binary value to output image array
                outputImg[row, col, band] = struct.unpack(dataFormat, data)[0]

    return outputImg


def main():
    # Required image information
    inputImgName = '../img/Pan.img'
    outputImgName = '../img/Pan.tif'
    numRow = 600*4
    numCol = 800*4
    numBand = 1
    dataFormat = 'H'        # 2-byte unsigned integer

    outputImg = readBsq(inputImgName, numRow, numCol, numBand, dataFormat)
    for i in range(numBand):
        # Determine the target value range after enhancement
        hist, _ = np.histogram(outputImg[:, :, i], bins=2048)
        accu = hist.cumsum()
        pc = 100. * accu / hist.sum()   # Cumulative probability array
        idxMin = (np.abs(pc - 2)).argmin()      # Index of %2
        idxMax = (np.abs(pc - 98)).argmin()     # Index of %98

        # Max/min value of output image
        valMin = np.where(accu == accu[idxMin])[0][-1]
        valMax = np.where(accu == accu[idxMax])[0][0]

        # Compute parameters of linear transform function
        slope = 255.0 / (valMax - valMin)
        incpt = 0 - valMin * slope

        # Update band values
        outputImg[:, :, i] = outputImg[:, :, i] * slope + incpt

    # Limit the output image values
    outputImg = np.clip(outputImg, 0, 255)

    # Show the result
    cv2.imshow(outputImgName, outputImg[:, :, :3].astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyWindow(outputImgName)

    # Save image
    cv2.imwrite(outputImgName, outputImg[:, :, :3].astype(np.uint8))


if __name__ == '__main__':
    main()
