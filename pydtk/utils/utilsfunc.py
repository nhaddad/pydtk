#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 17:57:25 2018

@author: nhaddad
"""
import numpy as np


def nextpow2(number):
    """
    Find 2^n that is equal to or less than number

    Syntax:
    nextpow2(n)

    ex: nextpow2(350)

    Utility function used by fft methods to compute the maximum array size
    which is power of 2

    """
    return int(np.log2(number))


def subwindowcoor(*coor, **kargs):
    """
    subwindow coordinate generator. Used to generate the coordinates of
    NWX x NWY subwindows in the area defined by *coor.
    Usage:
    subwin = subwindowcoor(200,300,500,600, NWX=2, NWY=2)
    divide de area defined by X=[200,300] and Y=[500,600] in 4 subareas
    subwin can be used in a for loop to give one by one all the coordinates
    and the index i,j of the coordinate

    for i,j,xi,xf,yi,yf in subwin:
        print i,j,xi,xf,yi,yf

    will output:
    0 0 200 250 500 550
    0 1 250 300 500 550
    1 0 200 250 550 600
    1 1 250 300 550 600



    """
    nwx = kargs.get('NWX', 10)  # set number of windows in x direction
    nwy = kargs.get('NWY', 10)  # set number of windows in y direction

    x1, x2, y1, y2 = coor

    wx = (x2-x1)//nwx  # size of every subwindow in x
    wy = (y2-y1)//nwy  # size of every subwindow in y

    for i in range(nwx):
        for j in range(nwy):
            if i == 0:
                xi = x1 + i * wx

            else:
                xi = x1 + i * wx + 1

            if j == 0:
                yi = y1 + j * wy
            else:
                yi = y1 + j * wy + 1

            xf = x1 + (i + 1) * wx  # - 1
            yf = y1 + (j+1) * wy  # - 1
            yield(i, j, xi, xf, yi, yf)


def genspectralimits(image, x0, y0, width, angle=1.518):
    """
    Generator that yields coordinates of the tilted spectra to be
    used for spectra  extraction
    x0, y0: coordinates of lower left point of the line that pass through the left side
    of the slit
    width: width in pixels of the slit
    angle: angle in radians obtained by spectralangle function or computed by hand


    example:  limits = genspectrallimits(1000,0,30, angle=1.52)

    TODO: modify this in such a way that uses a flat field image to detect
    the position of the spectral lines

    TODO: move it to "utils"

    """

    for y in range(image.shape[0]):
        x = int((1/np.tan(angle))*(y - y0) + x0)
        yield y, x, x + width
