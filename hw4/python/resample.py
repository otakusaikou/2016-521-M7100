#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module is for image resampling."""
import numpy as np
import os
from scipy.misc import imread
from scipy.misc import imsave
from sympy import solve
from sympy import symbols
import sys
from trans8param import get8params


def getValue(img, x, y):
    """Bilinear interpolation."""
    # Get coordinates of nearest four points
    x0 = np.floor(x)
    x1 = x0 + 1
    y0 = np.floor(y)
    y1 = y0 + 1

    # Ensure the coordinates of four points are in the right image extent
    x0 = int(np.clip(x0, 0, img.shape[1] - 1))
    x1 = int(np.clip(x1, 0, img.shape[1] - 1))
    y0 = int(np.clip(y0, 0, img.shape[0] - 1))
    y1 = int(np.clip(y1, 0, img.shape[0] - 1))

    # Get intensity of nearest four points
    Ia = img[y0, x0]  # Upper left corner
    Ib = img[y1, x0]  # Lower left corner
    Ic = img[y0, x1]  # Upper right corner
    Id = img[y1, x1]  # Lower right corner

    # Compute the weight of four points
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def getBound(width, height, param, shiftxy=(0, 0), shiftXY=(0, 0)):
    """Compute the bound of image in target CRS."""
    boundXY = np.array(
        [[0, 0], [width, 0], [width, -height], [0, -height]]).astype(np.double)
    boundXY[:, 0] -= shiftXY[0]
    boundXY[:, 1] -= shiftXY[1]

    a, b, c, d, e, f, g, h = param
    x, y, X, Y = symbols("x y X Y")
    F = [(a * x + b * y + c) / (g * x + h * y + 1) - X,
         (d * x + e * y + f) / (g * x + h * y + 1) - Y]
    sol = solve(F, x, y)

    boundxy = []
    for col, row in boundXY:
        boundxy.append(
            [sol[x].evalf(subs={X: col, Y: row}) + shiftxy[0],
             sol[y].evalf(subs={X: col, Y: row}) + shiftxy[1]])
    boundxy = np.array(boundxy)

    xmin = np.double(boundxy[:, 0].min())
    xmax = np.double(boundxy[:, 0].max())
    ymin = np.double(boundxy[:, 1].min())
    ymax = np.double(boundxy[:, 1].max())

    return xmin, xmax, ymin, ymax


def main():
    # Define file names
    controlPtFileName = 'cp_MS.txt'
    imageName = '../img/MS.tif'
    # controlPtFileName = 'cp_Pan.txt'
    # imageName = '../img/Pan.tif'

    # Read control point information from file
    controlPts = np.genfromtxt(controlPtFileName, dtype=[
        ('Name', 'S10'), ('C', 'f8'), ('R', 'f8'),      # Image CRS
        ('E', 'f8'), ('N', 'f8')], skip_header=1)       # TWD97 CRS

    # Read image information
    img = imread(imageName)
    if len(img.shape) == 3:     # For multi-spectral image
        height, width, numBand = img.shape
    else:                       # For panchromatic image
        height, width = img.shape
        numBand = 1

    CR = controlPts['C'], height - controlPts['R']
    EN = controlPts['E'], controlPts['N']
    shiftCR = CR[0].mean(), CR[1].mean()
    shiftEN = EN[0].mean(), EN[1].mean()

    # Compute 8 parameters
    param, paramStd = get8params(EN, CR, shiftEN, shiftCR)

    # Get extent of the image after transformation
    xmin, xmax, ymin, ymax = getBound(width, height, param, shiftEN, shiftCR)

    # Generate grid points
    rangeX = np.linspace(xmin, xmax, num=width) - shiftEN[0]
    rangeY = np.linspace(ymax, ymin, num=height) - shiftEN[1]

    a, b, c, d, e, f, g, h = param

    # Generate new image sampling
    resImg = np.zeros((height * width, numBand))
    idx = 0
    curValue = 0    # Current percentage of completion
    pxNum = height * width
    sys.stdout.write("Processing... %3d%%" % 0)

    for y in rangeY:
        for x in rangeX:
            X = (a * x + b * y + c) / (g * x + h * y + 1) + shiftCR[0]
            Y = -((d * x + e * y + f) / (g * x + h * y + 1) + shiftCR[1])
            val = getValue(img, X, Y)
            resImg[idx, :] = val[0, :]
            idx += 1

        # Update the percentage of completion
        if curValue < int(100.0 * idx / pxNum):
            curValue = int(100.0 * idx / pxNum)
            sys.stdout.write("\b" * 4)
            sys.stdout.write("%3d%%" % curValue)
            sys.stdout.flush()
    sys.stdout.write("\n")

    # Save result
    if numBand == 1:
        resImg = resImg.reshape(height, width)
    else:
        resImg = resImg.reshape(height, width, numBand)
    fname, ext = os.path.splitext(imageName)
    imsave("../images/" + fname + "_modified" + ext, resImg.astype(np.uint8))

    return 0


if __name__ == '__main__':
    main()
