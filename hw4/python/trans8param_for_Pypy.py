#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
"""Perform projective transformation."""
import numpy as np


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def get8params(xy, XY, shiftxy=(0, 0), shiftXY=(0, 0)):
    """Compute 8 parameters with given control points."""
    # Shift the data to prevent error caused by large numbers
    x = xy[0] - shiftxy[0]
    y = xy[1] - shiftxy[1]
    X = XY[0] - shiftXY[0]
    Y = XY[1] - shiftXY[1]

    numPt = len(x)

    # Compute coefficient and constant matrices
    B = np.matrix(np.zeros((2 * numPt, 8))).astype(np.double)
    B[:numPt, :3] = np.dstack((x, y, np.ones(numPt))).reshape(-1, 3)
    B[numPt:, 3:6] = np.dstack((x, y, np.ones(numPt))).reshape(-1, 3)
    B[:numPt, 6:] = -np.dstack(((x * X), (y * X))).reshape(-1, 2)
    B[numPt:, 6:] = -np.dstack(((x * Y), (y * Y))).reshape(-1, 2)

    f = np.matrix(np.concatenate((X, Y))).T

    # Solve unknown parameters
    N = np.matrix(B.T.dot(B))       # Compute normal matrix
    t = np.matrix(B.T.dot(f))       # Compute t matrix
    dX = N.I.dot(t)

    # Error assessment
    V = (B * dX) - f
    s0 = np.sqrt((V.T * V) / (B.shape[0] - B.shape[1]))[0, 0]

    SigmaXX = s0**2 * N.I
    paramStd = np.sqrt(np.diag(SigmaXX))

    return dX, paramStd


def transPts(xy, param, shiftxy=(0, 0), shiftXY=(0, 0)):
    """Perform projective transformation."""
    x = xy[0] - shiftxy[0]
    y = xy[1] - shiftxy[1]
    a, b, c, d, e, f, g, h = param
    X = (a * x + b * y + c) / (g * x + h * y + 1) + shiftXY[0]
    Y = (d * x + e * y + f) / (g * x + h * y + 1) + shiftXY[1]

    return X, Y


def main():
    # Define file names
    controlPtFileName = 'cp_MS.txt'
    inputPtFileName = 'chkp_MS.txt'
    # controlPtFileName = 'cp_Pan.txt'
    # inputPtFileName = 'chkp_Pan.txt'

    # Read control point information from file
    controlPts = np.genfromtxt(controlPtFileName, dtype=[
        ('Name', 'S10'), ('C', 'f8'), ('R', 'f8'),      # Image CRS
        ('E', 'f8'), ('N', 'f8')], skip_header=1)       # TWD97 CRS

    CR = map(lambda key: controlPts[key], list("CR"))
    EN = map(lambda key: controlPts[key], list("EN"))
    shiftCR = CR[0].mean(), CR[1].mean()
    shiftEN = EN[0].mean(), EN[1].mean()

    # Read the points to be transformed
    inputPts = np.genfromtxt(inputPtFileName, dtype=[
        ('Name', 'S10'), ('C', 'f8'), ('R', 'f8'),      # Image CRS
        ('E', 'f8'), ('N', 'f8')], skip_header=1)       # TWD97 CRS

    CR2 = map(lambda key: inputPts[key], list("CR"))
    EN2 = map(lambda key: inputPts[key], list("EN"))

    # Compute 8 parameters
    param, paramStd = get8params(CR, EN, shiftCR, shiftEN)

    # Transform the input points
    EN3 = transPts(CR2, param, shiftCR, shiftEN)

    # Compute the RMS of check points
    RMSX = np.sqrt(np.power(EN3[0] - EN2[0], 2).mean())
    RMSY = np.sqrt(np.power(EN3[1] - EN2[1], 2).mean())
    print("RMSX = %.8f") % RMSX
    print("RMSY = %.8f") % RMSY

    return 0


if __name__ == '__main__':
    main()
