#!/usr/bin/python
# Filename: dtk.py

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 08:55:44 2014

@author: nhaddad

nhaddad@eso.org
nhaddads@gmail.com

Defines the class Image and creates some methods to make statistic computation over the images

TODO: Use a list of files name to allow changng names, removing files



"""


from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import datetime
from .utils.utilsfunc import nextpow2
from .utils.utilsfunc import subwindowcoor
# from .utils.utils import genspectralimits


"""
Define Module Exceptions
"""


class ObjectNotAnImage(Exception):
    '''
    TBC
    '''
    pass


class LenTooShort(Exception):
    '''
    TBC
    '''
    pass


class ObjectNotAnString(Exception):
    '''
    TBC
    '''
    pass


class Image(object):
    '''
    Image class implements the Image object. This object has the following methods:

    def __init__(self, filename=None, ext=0, slice=0, xps=0, xos=0, yps=0, yos=0, **kargs):
    def findheaderkeyword(self, wildcard):
    def findheadercomment(self, wildcard):
    def transpose(self):
    def rot(self, angle=90):
    def fliplr(self):
    def flipud(self):
    def filenameinfo(self, **kargs):
    def __getitem__(self, key):
    def crop(self, xi, xf, yi, yf):
    def copy(self):
    def __add__(self, param):
    def __sub__(self, param):
    def __mul__(self, param):
    def __truediv__(self, param):
        # def __div__(self,param):
    def medianfilt(self, kernel_size=3):
    def mean(self, *coor, **kargs):
    def median(self, *coor, **kargs):
    def var(self, *coor, **kargs):
    def std(self, *coor, **kargs):
    def mask(self, NSTD=3.0, *coor, **kargs):
    def save(self, filename):
    def display(self, *window, **kargs):
    def hist(self, *window, **kargs):
    def rowstacking(self, *xcoor, **kargs):
    def columnstacking(self, *ycoor, **kargs):
    def get_xcoor(self, *coor):
    def get_ycoor(self, *coor):
    def get_windowcoor(self, *coor):
    def stat(self, *coor, **kargs):
    def extractspect(self, x0, y0, width, angle):
    def avecol(self, *ycoor, **kargs):
    def mediancol(self, *ycoor, **kargs):
    def averow(self, *xcoor, **kargs):
    def medianrow(self, *xcoor, **kargs):
    def set_value(self, value=1000, *coor, **kargs):
    def get_data(self):
    def get_xi(self):
    def get_xf(self):
    def get_yi(self):
    def get_yf(self):
    def set_xps(self, xps):
    def get_xps(self):
    def set_xos(self, xos):
    def get_xos(self):
    def set_yps(self, yps):
    def get_yps(self):
    def set_yos(self, yos):
    def get_yos(self):
    def format(self):
    def subwindows(self, nx=10, ny=10):
    def fftcol(self, ReadOutSpeed=100, Vdelay=5, ColStart=None, ColEnd=None, **kargs):
    def fftrow(self, ReadOutSpeed=100, RowStart=None, RowEnd=None, **kargs):
    def fft2D(self, **kargs):
    def bitcounts(self, *coor):
    def addnormalnoise(self, stddev):
    def addsinusoidalnoise(self, fnvector, ampvector, rofreq, vdelay, **kargs):

    '''

    def __init__(self, filename=None, ext=0, slice=0, xps=0, xos=0, yps=0, yos=0, **kargs):
        """
        Initializes Image instance.
        If Image is called with another image as a parameter, then just copy all the attributes,
        including the data.
        This option to call the Image class is used inside the redefinition of the arithmetic
        operations on images (+,-,*,/)

        This class can be used to generate a simulated image if filename==None, the possible
        options are:
        NROWS: Number of rows for simulated image
        NCOLS: Number of columns for simulated image
        CF: conversion factor in e/ADU
        RON: read out noise in electrons
        BIAS: bias level in ADUs
        FLUX: counts per sec in ADU
        DIT: integration time
        SHADOW: generate a shadowing on the detector image, SHADOW is a float value
        from 0 to 100 (percentage)
        FPN_P: Fix pattern noise matrix to apply toeach pixel in the detector
        FPN_C: Column Fix pattern noise vector to apply to all columns (normally IR sensors
        suffers this FPN)
        FPN_R: Row Fix pattern noise vector to apply to all rows  (normally IR sensors
        suffers this FPN)

        Syntax:
        im=Image('filename.fits',ext[,NAME=True]) ==> creates an image from a fits file
        im2=Image(im)    ==>make a copy of an existing Image object (im in this case)
        im3=Image('filename.fits',cube=True) => read the image cube
        im4=Image('filename.fits',cslice=5) => i a data cube, read the 6th slice
        im4=Image(None, NROWS=512, NCOLS=512)
        im5=Image(None, NROWS=512, NColS=512, CF=2, RON=3, BIAS=1000, FLUX=100, DIT=10)
            CF=2e/ADU, RON=3e, BIAS=1000ADU, FLUX=100 ADU/sec, DIT=10sec
        im6=Image(None, NROWS=512, NColS=512, CF=2, RON=3, BIAS=1000, FLUX=100,
                  DIT=10, SHADOWC= 10, FPN_P=fpnp) fpnp = generate_fpn_row(VALUE=0.01)

        How the simulated image is generated:
        * Create an array with mean value of BIAS and normal random amplitude of RON/CF
        * if DIT and FLUX are > 0, add DIT*FLUX ADUs + noise of amplitude sqrt(DIT*FLUX/CF)
        * if SHADOWC >0, apply a sinusoidal function  to columns, the peak to peak amplitud of
            the modulation
        * if SHADOWR >0, apply a sinusoidal function  to rows, the peak to peak amplitud of
            the modulation


        * if FPN_P and/or FPN_C and/or FPN_R is provided, then apply them to the detector image.
        They correspond to fix pattern noise
        """

        if isinstance(filename, Image):  # if filename is an instance of Image... Copy attributes
            self.ext = filename.ext
            self.filename = filename.filename
            self.path = filename.path
            self.header = filename.header
            self.data = filename.data
            self.slice = filename.slice
            self.xf = filename.xf
            self.xi = filename.xi
            self.yf = filename.yf
            self.yi = filename.yi
            self.shape = (self.xf, self.yf)
            self.xps = filename.xps
            self.xos = filename.xos
            self.yps = filename.yps
            self.yos = filename.yos

        elif filename is None:  # if filename is None => create a syntetic image
            self.ext = 0
            self.filename = 'Simulated'
            self.path = ''
            self.header = None
            self.slice = None
            self.xf = kargs.get('NROWS', 512)
            self.xi = 0
            self.yf = kargs.get('NCOLS', 512)
            self.yi = 0
            self.data = np.zeros((self.xf, self.yf))
            self.shape = (self.xf, self.yf)
            # check CF, if not given, defaul to 2 [e/ADU]
            cf = kargs.get('CF', 2.0)
            # check if Flux (photons/sec) different from zero
            flux = kargs.get('FLUX', 0)
            # check if DIT (sec) diferent from zero
            dit = kargs.get('DIT', 0)
            # check if RON (e) diferent from zero
            ron = kargs.get('RON', 3)
            # check if BIAS (ADU) diferent from zero
            bias = kargs.get('BIAS', 1000)

            # generate an array of NROWSxNCOLS and fill it with (flux/cf) +/- sqrt(flux/cf)
            # if DIT>0 and FLUX>0
            if (dit > 0 and flux > 0):
                signal_amplitud = dit*flux  # in electrons
                shotnoise_amplitud = np.sqrt(signal_amplitud)  # in electrons
                # Generate array filled with "signal"
                signal = np.empty((self.xf, self.yf))
                signal.fill(signal_amplitud)
                signal = signal/cf  # to convert to ADUs
                self.data = signal.copy()  # NEW

                # generate array filled with "shotnoise"
                # shotnoise = np.random.normal(signal_amplitud, shotnoise_amplitud, self.xf*self.yf)
                # in electrons
                # shotnoise = shotnoise - signal_amplitud  # Leave only the shotnoise
                shotnoise = np.random.normal(0, shotnoise_amplitud/cf, self.xf*self.yf)  # in ADUs
                shotnoise = shotnoise.reshape((self.xf, self.yf))

                self.data = self.data + shotnoise

            # Initialize shadow and FPN matrix with all elements in 1.0
            shadow_coef = np.empty((self.xf, self.yf))
            shadow_coef.fill(1.0)

            FPNTotal = np.empty((self.xf, self.yf))
            FPNTotal.fill(1.0)  # initialize FPNTotal with 1.0

            # Add shadow to image if there is shadow in row and/or columns
            # check if SHADOW!=0, if not, shadow is a percentage shadow factor
            shadowr = kargs.get('SHADOWR', 0)
            shadowc = kargs.get('SHADOWC', 0)
            if (shadowr != 0 or shadowc != 0) and (dit > 0 and flux > 0):
                factorr = shadowr/100.0  # express shadow in porcentage
                factorc = shadowc/100.0
                # generate the row multiplication coeficients using a sine [0..1]
                rowshadow = (np.sin(np.linspace(0, np.pi, self.xf)).reshape((self.xf, 1)))*factorr
                # colshadow = np.sin(np.linspace(0,np.pi,self.yf))*factorc  #column coeficients multiplied by sine[0..1]
                colshadow = (np.sin(np.linspace(0, np.pi, self.yf))).reshape(1, self.yf)*factorc

                # generate the matrix of multiplication coeficients
                shadow_coef = rowshadow + colshadow

                # shadow_shnoise_coef = np.sqrt(shadow_coef)

                # self.data =  signal*(1-factorr-factorc) + signal * shadow_coef
                # data signal modulated by shadow
                self.data = self.data*(1-factorr-factorc) + self.data * shadow_coef
                # self.data =  signal * (1 - factorr -factorc + shadow_coef)

            # Add FPN if the FPN matrix were defined for pixel, column or row, or for all of them
            # check if FPNpixel not empty, must be same dimension as self.data
            FPNpixel = kargs.get('FPN_P', None)
            if isinstance(FPNpixel, np.ndarray):
                # check if dimensions of FPN array are correct
                if self.shape == FPNpixel.shape:
                    self.data = self.data * FPNpixel
                    FPNTotal = FPNTotal * FPNpixel  # TBC
                else:
                    print("Error: not same dimensions..")

            # check if FPNcol not empty, must be vector of dim col
            FPNcol = kargs.get('FPN_C', None)
            if isinstance(FPNcol, np.ndarray):
                if self.shape[1] == len(FPNcol):
                    self.data = self.data * FPNcol
                    FPNTotal = FPNTotal * FPNcol  # TBC
                else:
                    print("Error: not same dimensions..")

            # check if FPNrow not empty, must be vector of dim row
            FPNrow = kargs.get('FPN_R', None)
            if isinstance(FPNrow, np.ndarray):
                if self.shape[0] == len(FPNrow):
                    FPNrow = FPNrow.reshape((self.shape[0], 1))
                    self.data = self.data * FPNrow
                    FPNTotal = FPNTotal * FPNrow  # TBC
                else:
                    print("Error: not same dimensions..")

            # if DIT>0 => multiply the shadow coef by the total FPN and then take the sqrt to
            # multiply this by the original shot noise
            if dit > 0 and shadowr > 0 and shadowc > 0:
                totalshotcoef = np.sqrt(np.abs((1 - factorr - factorc + shadow_coef) * FPNTotal))
                self.data = self.data + totalshotcoef * shotnoise
            elif dit > 0 and shadowr == 0 and shadowc == 0:
                # totalshotcoef = np.sqrt(FPNTotal)
                # self.data = signal + totalshotcoef*shotnoise           #ORIGINAL
                self.data = self.data  # + shotnoise #totalshotcoef*shotnoise         #NEW

            # Finally add bias level  and RON here, not before
            # data = (np.random.standard_normal((self.xf,self.yf))*ron/cf) + bias
            data = np.random.normal(bias, ron/cf, self.xf*self.yf)  # in ADUs
            data = data.reshape((self.xf, self.yf))
            self.data = data + self.data

            # check if detector values are less or equal than 65535
            self.data = self.data.clip(min=0, max=65535)
            self.xps = 0
            self.xos = 0
            self.yps = 0
            self.yos = 0

        # if filename exist then read data
        elif filename != None:
            # Read image data
            # datos,hrd=pyfits.getdata(PATH2IMAGES+filename,chip,header=True)
            # print filename
            # load data and header information
            datos, hrd = pyfits.getdata(filename, ext, header=True)
            self.ext = ext
            self.header = hrd
            # check data is not cube
            if np.size(datos.shape) == 2:
                self.data = datos.astype('float32')
                # self.data = datos
                self.xf = datos.shape[0]
                self.yf = datos.shape[1]
                self.slice = None

            elif np.size(datos.shape) == 3 and datos.shape[0] == 1:
                self.data = datos[0, :, :].astype('float32')
                # self.data = datos[0,:,:]
                self.xf = datos.shape[1]
                self.yf = datos.shape[2]
                self.slice = 0
            elif np.size(datos.shape) == 3 and slice != None and slice < datos.shape[0]:
                self.data = datos[slice, :, :].astype('float32')
                # self.data = datos[slice,:,:]
                self.xf = datos.shape[1]
                self.yf = datos.shape[2]
                self.slice = slice
            """
            elif size(datos.shape) ==3 and datos.shape[0]>1 and slice==None:  #Load all the slices
                self.data = datos[:,:,:].astype('float32')
                # self.data = datos[slice,:,:]
                self.xf = datos.shape[1]
                self.yf = datos.shape[2]
                self.zf = datos.shape[0]
                self.slice = slice
            """

            self.filename = os.path.basename(filename)
            self.path = os.path.dirname(filename)
            self.xi = 0
            self.yi = 0
            # self.shape = self.data.shape
            self.shape = (self.xf, self.yf)
            self.xps = xps
            self.xos = xos
            self.yps = yps
            self.yos = yos

            if kargs.get('NAME', False):
                print(f'Loading {filename}...')

    def findheaderkeyword(self, wildcard):
        """
        Print out all the headers that have the wild card

        Syntax:
        im.findheaderkeyword('DAT')
        """

        for key in self.header.cards:
            if key[0].find(wildcard) != -1:
                print(f'{key[0]} : {key[1]} /{key[2]}')

    def findheadercomment(self, wildcard):
        """
        Print out all the headers that have the wild card in the comment

        Syntax:
        im.findheadercomment('ADU')
        """

        for key in self.header.cards:
            if key[2].find(wildcard) != -1:
                print(f'{key[0]} : {key[1]} /{key[2]}')

    def transpose(self):
        """
        Transpose data array

        im.transpose()

        """
        self.data = self.data.transpose()
        # set the correct xf, yf and shape after transposing the data array
        self.xf = self.data.shape[0]
        self.yf = self.data.shape[1]
        self.shape = (self.xf, self.yf)

    def rot(self, angle=90):
        """
        Rotate image data according to the requested angle. Angle must be 90, 180 or 270

        Usage:
        im.rot(90)
        """
        if angle == 90:
            self.data = np.rot90(self.data, 1)

        elif angle == 180:
            self.data = np.rot90(self.data, 2)

        elif angle == 270:
            self.data = np.rot90(self.data, 3)

        else:
            print("Angle must be 90, 180 or 270!")

        self.xf = self.data.shape[0]
        self.yf = self.data.shape[1]
        self.shape = (self.xf, self.yf)

    def fliplr(self):
        """
        Flip image in direction left-rigt

        Usage:
        im.fliplr()
        """
        self.data = np.fliplr(self.data)

    def flipud(self):
        """
        Flip image in direction up-down

        Usage:
        im.flipud()
        """
        self.data = np.flipud(self.data)

    def get_filename(self, **kargs):
        """
        Syntax:
        im.get_filename()
        name = im.get_filename(RETURN=True)

        Print or return the filename of the image
        """

        if kargs.get('RETURN', False):
            return self.filename
        else:
            print(f'File name: {self.filename}')

    def get_path(self, **kargs):
        """
        Syntax:
        im.get_path()
        name = im.get_path(RETURN=True)

        Print or return the path to directory of the image
        """

        if kargs.get('RETURN', False):
            return self.path
        else:
            print(f'Path to file: {self.path}')

    def get_extension(self, **kargs):
        """
        Syntax:
        im.get_extension()
        im.get_extension(RETURN=True)

        """
        if kargs.get('RETURN', False):
            return self.ext
        else:
            print(f'extension={self.ext}')

    def __getitem__(self, key):
        """
        This methods return the pixel value associated with position key[0],key[1]
        """

        try:
            # return self.data[(key[0],key[1])]
            return self.data[(key[0], key[1])]
        except IndexError:
            print('IndexError: Index out of range')

    def crop(self, xi, xf, yi, yf):
        """
        Syntax:
        c=im.crop(xi,xf,yi,yf)

        Generate a new image using a sub area of the original image

        Example:
        c=b1.crop(100,500,400,600)
        This creates an image of 400x200 pix, in which c is a subset of image b1.
        """

        im = Image(self)
        im.filename = self.filename
        im.data = self[xi:xf, yi:yf].copy()
        im.xf = im.data.shape[0]
        im.yf = im.data.shape[1]
        im.xi = 0
        im.yi = 0
        im.shape = (im.xf, im.yf)
        return im

    def copy(self):
        """
        Syntax:
        c=im.copy()

        Generate a new image using  of the original image

        Example:
        c=b1.copy()
        This creates a copy of image b1.
        """

        im = Image(self)
        im.filename = self.filename
        im.header = self.header
        im.data = self[:, :].copy()
        im.xf = im.data.shape[0]
        im.yf = im.data.shape[1]
        im.xi = 0
        im.yi = 0
        im.shape = (im.xf, im.yf)
        return im

    def __add__(self, param):
        """
        Syntax:
        a = b+c
        creates a new image 'a' which is the addition pixel by pixel of images b and c

        Redifine the '+' operation for 2 images or one image and one number
        """
        # if self and param are Images
        if isinstance(self, Image) and isinstance(param, Image):
            a = self.data
            b = param.data
            im = Image(self)
            im.filename = self.filename+'+'+param.filename
        # if self is Image and param is a number or an ndarray
        elif isinstance(self, Image) and isinstance(param, (int, float, np.float32, np.dnarray)):
            a = self.data
            b = param
            im = Image(self)  # create empty image container
            im.filename = self.filename+'+'+str(param)
        elif isinstance(self, Image) and isinstance(param,  np.ndarray):
            a = self.data
            im = Image(self)
            b = param
            if b.ndim == 1:
                if b.shape[0] == im.shape[0]:
                    b = b.reshape(b.shape[0], 1)
                elif b.shape[0] == im.shape[1]:
                    b = b.reshape(1, b.shape[0])

        # perform operation
        im.data = a + b
        # im.filename=''
        return im

    def __sub__(self, param):
        """
        Syntax:
        a = b-c
        creates a new image a which is the subtraction pixel by pixel of images b and c

        Redifine the '-' operation for 2 images or one image and one number
        """

        if isinstance(self, Image) and isinstance(param, Image):
            a = self.data
            b = param.data
            im = Image(self)
            im.filename = self.filename+'-'+param.filename
        elif isinstance(self, Image) and isinstance(param, (int, float, np.float32)):
            a = self.data
            b = param
            im = Image(self)
            im.filename = self.filename+'-'+str(param)
        elif isinstance(self, Image) and isinstance(param,  np.ndarray):
            a = self.data
            im = Image(self)
            b = param
            if b.ndim == 1:
                if b.shape[0] == im.shape[0]:
                    b = b.reshape(b.shape[0], 1)
                elif b.shape[0] == im.shape[1]:
                    b = b.reshape(1, b.shape[0])
        im.data = a - b
        # im.filename=''
        return im

    def __mul__(self, param):
        """
        Syntax:
        a = b*c
        creates a new image a which is the product pixel by pixel of images b and c

        Redifine the '*' operation for 2 images or one image and one number
        """

        if isinstance(self, Image) and isinstance(param, Image):
            a = self.data
            b = param.data
            im = Image(self)
            im.filename = '('+self.filename+'*'+param.filename+')'
        elif isinstance(self, Image) and isinstance(param, (int, float, np.float32)):
            a = self.data
            b = param
            im = Image(self)
            im.filename = '('+self.filename+'*'+str(param)+')'
        elif isinstance(self, Image) and isinstance(param,  np.ndarray):
            a = self.data
            im = Image(self)
            b = param
            if b.ndim == 1:
                if b.shape[0] == im.shape[0]:
                    b = b.reshape(b.shape[0], 1)
                elif b.shape[0] == im.shape[1]:
                    b = b.reshape(1, b.shape[0])

        im.data = a*b
        # im.filename=''
        return im

    def __truediv__(self, param):
        # def __div__(self,param):
        """
        Syntax:
        a = b/c
        creates a new image a which is the quotient pixel by pixel of images b and c

        Redifine the '/' operation for 2 images or one image and one number
        Note: if the result  of the division is  inf or -inf or nan, then replace those values by 0.0
        """
        # both images
        if isinstance(self, Image) and isinstance(param, Image):
            a = self.data
            b = param.data
            im = Image(self)
            im.filename = '('+self.filename+'/'+param.filename+')'
        # image and integer or float
        elif isinstance(self, Image) and isinstance(param, (int, float, np.float32)):
            a = self.data
            b = param
            im = Image(self)
            im.filename = '('+self.filename+'/'+str(param)+')'
        elif isinstance(self, Image) and isinstance(param,  np.ndarray):
            a = self.data
            im = Image(self)
            b = param
            if b.ndim == 1:
                if b.shape[0] == im.shape[0]:
                    b = b.reshape(b.shape[0], 1)
                elif b.shape[0] == im.shape[1]:
                    b = b.reshape(1, b.shape[0])

        # if divisor has 0, then replace 'inf' or -inf or nan  by 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            im.data = a/b
        im.data[np.isinf(im.data) | np.isneginf(im.data) | np.isnan(im.data)] = 0.0

        return im

    def medianfilt(self, kernel_size=3):
        """
        Syntax:
        im_filtered = im.medianfilt(kernel_size=5)

        returns a new image which is the original but after applying
        signal.medfilt2d method

        TODO: check if it's wise to have this method that depends on scipy...
        """

        im = Image(self)  # create empty image container
        im.filename = self.filename
        im.data = self.data[:, :].copy()
        im.data = signal.medfilt2d(im.data, kernel_size)
        return im


# TODO :add **kargs to compute variance on one axis
    def mean(self, *coor, **kargs):
        """
        Syntax:
        im.mean(xi,xf,yi,yf)

        Computes the mean value for the image. If no coordinates are specified, the full image is used
        """
        axis = kargs.get('AXIS', None)
        Xi, Xf, Yi, Yf = self.get_windowcoor(*coor)

        return np.mean(self[Xi:Xf, Yi:Yf], axis=axis)

# TODO :add **kargs to compute variance on one axis
    def median(self, *coor, **kargs):
        """
        Syntax:
        im.median(xi,xf,yi,yf)

        Computes the median value for the image. If no coordinates are specified, the full image is used
        """
        axis = kargs.get('AXIS', None)
        Xi, Xf, Yi, Yf = self.get_windowcoor(*coor)

        return np.median(self[Xi:Xf, Yi:Yf], axis=axis)
# TODO :add **kargs to compute variance on one axis

    def var(self, *coor, **kargs):
        """
        Syntax:
        im.var(xi,xf,yi,yf) compute variance of complete array
        im.var(xi,xf,yi,yf, AXIS=0) compute variance of rows
        im.var(xi,xf,yi,yf, AXIS=1) compute variance of columns

        Computes the variance for the image. If no coordinates are specified,
        the full image is used
        """
        axis = kargs.get('AXIS', None)
        Xi, Xf, Yi, Yf = self.get_windowcoor(*coor)

        return np.var(self[Xi:Xf, Yi:Yf], axis=axis)

    def std(self, *coor, **kargs):
        """
        Syntax:
        im.std(xi,xf,yi,yf) compute std_dev of complete array
        im.std(xi,xf,yi,yf, AXIS=0) compute std_dev of rows
        im.std(xi,xf,yi,yf, AXIS=1) compute std_dev of columns

        Computes the standard deviation value for the image. If no coordenates
        are specified, the full image is used.
        If axis is 0 or 1, it will return a vector instead of a single number
        """
        axis = kargs.get('AXIS', None)
        Xi, Xf, Yi, Yf = self.get_windowcoor(*coor)

        return np.std(self[Xi:Xf, Yi:Yf], axis=axis)

    def mask(self, NSTD=3.0, *coor, **kargs):
        """
        Syntax:
        im.mask() generate mask of all pixels outside 3*std_dev from mean
        im.mask(NSTD=5.0)generate masked array using mean (full image) and 5*std_dev
        im.mask(NSTD=2.0,100,400,500,800) generate masked array of image using the mean
        and 2*std_dev computed at [100:400,500:800]

        Converts the image data into masked array with all the values +/- FACTOR*std masked out
        """
        # Computer mean value of image
        immean = self.mean(*coor)
        # compute std_dev for image
        imstd = self.std(*coor)
        self.data = np.ma.masked_outside(self.data, immean-NSTD*imstd, immean+NSTD*imstd)
        if kargs.get('COUNTS', False):
            pixels = self.shape[0]*self.shape[1]
            non_masked = self.data.count()
            print(f'Total pixels = {pixels}, masked pixels={pixels - non_masked}')

    def save(self, filename):
        """
        Save image in FITS format, and copy the header from the
        original file fits

        im.save('filename.fits')
        """
        hdu = pyfits.PrimaryHDU(self.data.astype(np.uint16), header=self.header)
        # hdu.header=self.header
        hdu.writeto(filename)

    def display(self, *window, **kargs):
        """
        Syntax:
        im.display(100, 600, 200, 800, vmin=190.0 ,vmax=201.5)

        Computes the mean value of the image, the standard deviation and then display the image with
        a vmin=mean-3*std and vmax=mean+3*std if no option MIN and/or MAX is given
        The cut options are: vmin, vmax
        ex:b1.display(vmin=190.0,vmax=201.5)
           b1.display(100,400,200,600) display window[100:400,200:600]

        hsize = Horizontal size of canvas
        vsize = Vertical size of canvas
        vmin  = low cut
        vmax  = high cut
        cmap  = colormap  ('jet', 'hot' 'gray')
        colorbar: True/False

        """

        Xi, Xf, Yi, Yf = self.get_windowcoor(*window)
        meanval = self.data[Xi:Xf, Yi:Yf].mean()
        stddev = self.data[Xi:Xf, Yi:Yf].std()

        # print(meanval, stddev)  # DEBUG

        if (meanval-stddev) > 0:
            lowercut = meanval - stddev
        else:
            lowercut = 0

        if (meanval+stddev) < 65535:
            uppercut = meanval + stddev
        else:
            uppercut = 65535
        # TODO Check min is greater than 0
        hsize = kargs.get('hsize', 8)
        vsize = kargs.get('vsize', 8)
        vmin = kargs.get('vmin', lowercut)
        vmax = kargs.get('vmax', uppercut)
        cmap = kargs.get('cmap', plt.get_cmap('jet_r'))
        if cmap == 'jet':
            cmap = plt.get_cmap('jet_r')
        elif cmap == 'hot':
            cmap = plt.get_cmap('hot')
        elif cmap == 'gray':
            cmap = plt.get_cmap('gray')

        # clf()
        plt.figure(figsize=(vsize, hsize))
        plt.title(self.filename)

        plt.imshow(self[Xi:Xf, Yi:Yf], vmin=vmin, vmax=vmax,
                   interpolation='nearest', cmap=cmap, origin='lower')

        print(f'Cut levels={vmin} and {vmax}')

        plt.xlabel('Ycoor')
        plt.ylabel('Xcoor')
        if kargs.get('colorbar', True):
            plt.colorbar()  # plt.colorbar()

        plt.show()  # plt.show()

    def hist(self, *window, **kargs):
        """
        Plot an histogram of the image provided window (full image if not provided)

        im.hist()
        im.hist(LOG=False)  Plot histogram in linear scale
        im.hist(MIN=200, MAX=500)  Make histogram in log scale (default) for values between 200 and 500
        im.hist(BINS=50)  Make histogram using 50 bins
        kargs:
        LOG: True/False to plot in log scale or linear scale
        MIN: minimum value to start the histogram
        MAX: maximum value to consider in the histogram
        BINS: number of bins to use
        CUMULATIVE : Make a cumulative histogram
        RETURN: if True it returns a histogram array and an edge array
        Note: to plot the returned histogram you can use plt.bar(x[:-1],y, log=1)  x is the edge array and y is the hist freq

        """
        Xi, Xf, Yi, Yf = self.get_windowcoor(*window)

        MIN = kargs.get('MIN', self.data.min())
        MAX = kargs.get('MAX', self.data.max())
        BINS = kargs.get('BINS', 256)
        CUMULATIVE = kargs.get('CUMULATIVE', False)
        logscale = kargs.get('LOG', True)
        if kargs.get('RETURN', False):
            return np.histogram(self.data[Xi:Xf, Yi:Yf].flatten(), bins=BINS, range=(MIN, MAX))

        else:
            plt.clf()
            plt.hist(self.data[Xi:Xf, Yi:Yf].flatten(), bins=BINS,
                     log=logscale, range=(MIN, MAX), cumulative=CUMULATIVE)
            plt.show()

    def rowstacking(self, *xcoor, **kargs):
        """
        Syntax:
        im.rowstacking(xi,xf,MIN=190,MAX=250)

        Plot rows between coordinates yi and yf, with Y scale range going from
        190 ADUs till 250 ADUs

        """

        x1, x2 = self.get_xcoor(*xcoor)

        plt.figure()

        plt.plot(self.get_data().transpose()[self.yi:self.yf, x1:x2], 'b.', ms=2.0)
        plt.grid(True)
        plt.title('Row Stacking from [%d:%d]' % (x1, x2))
        plt.xlabel('Column')
        plt.ylabel('Signal [ADUs]')

        ymax = kargs.get('MAX', self.data.max())
        ymin = kargs.get('MIX', self.data.min())
        plt.ylim(ymin, ymax)

        plt.show()

    def columnstacking(self, *ycoor, **kargs):
        """
        Syntax:
        im.columnsstacking(yi,yf,MIN=190,MAX=250)

        Plot columns between coordinates xi and xf, with Y scale range going from
        190 ADUs till 250 ADUs

        """

        y1, y2 = self.get_ycoor(*ycoor)

        plt.figure()

        plt.plot(self[self.xi:self.xf, y1:y2], 'b.', ms=2.0)
        plt.grid(True)
        plt.title('Column Stacking from [%d:%d] ' % (y1, y2))
        plt.xlabel('Row')
        plt.ylabel('Signal [ADUs]')

        ymax = kargs.get('MAX', self.data.max())
        ymin = kargs.get('MIN', self.data.min())
        plt.ylim(ymin, ymax)

        plt.show()

    def get_xcoor(self, *coor):
        """
        To be completed
        """
        if not coor:
            Xi = self.xi
            Xf = self.xf

        elif len(coor) == 1:
            Xi = coor[0]
            Xf = self.xf
        elif len(coor) == 2:
            Xi = coor[0]
            Xf = coor[1]

        # TODO: check if Xf>Xi and Yf>Yi and also that the values are no negatives and no greater
        #      than self.xf and self.yf
        return Xi, Xf

    def get_ycoor(self, *coor):
        """

        """
        if not coor:
            Yi = self.yi
            Yf = self.yf

        elif len(coor) == 1:
            Yi = coor[0]
            Yf = self.yf
        elif len(coor) == 2:
            Yi = coor[0]
            Yf = coor[1]

        # TODO: check if Xf>Xi and Yf>Yi and also that the values are no negatives and no greater
        #      than self.xf and self.yf
        return Yi, Yf

    def get_windowcoor(self, *coor):
        """
        Computes the window coordinates on the detector according to the values entered through the
        coor tuple

        """

        if not coor:  # No tuple => use full chip
            Xi = self.xi
            Xf = self.xf
            Yi = self.yi
            Yf = self.yf
        elif len(coor) == 1:  # Only one coordinate => assign to Xi
            Xi = coor[0]
            Xf = self.xf
            Yi = self.yi
            Yf = self.yf
        elif len(coor) == 2:  # Two coordinates => assign to Xi, Xf
            Xi = coor[0]
            Xf = coor[1]
            Yi = self.yi
            Yf = self.yf
        elif len(coor) == 3:  # Three coordinates => assign to Xi, Xf, Yi
            Xi = coor[0]
            Xf = coor[1]
            Yi = coor[2]
            Yf = self.yf
        elif len(coor) == 4:  # Four coordinates => assign to Xi, Xf, Yi, Yf
            Xi = coor[0]
            Xf = coor[1]
            Yi = coor[2]
            Yf = coor[3]

        # check if Xf>Xi and Yf>Yi and also that the values are no negatives and no greater
        # than self.xf and self.yf
        if Xi < self.xi or Xi > self.xf:
            Xi = self.xi
        if Xf < self.xi or Xf > self.xf:
            Xf = self.xf
        if Xi > Xf:
            Xi = Xf
        if Yi < self.yi or Yi > self.yf:
            Yi = self.yi
        if Yf < self.yi or Yf > self.yf:
            Yf = self.yf
        if Yi > Yf:
            Yi = Yf

        return Xi, Xf, Yi, Yf

    def stat(self, *coor, **kargs):
        """
        Syntax:
        im.stat(xi,xf,yi,xf[,NWX=10][,NWY=10][,SHOW=True][,SAVE=True][,TITLE='GraphTitle'][,LOG=True][,NSTD=6][FACTOR=True])

        Perform statistic analysis of window, if no coordinates are given, then perform
        statistic over whole image area. The following information is printed and plotted:
        mean
        median
        std_dev
        max
        min
        and a histogram of the pixel values is drawn
        options:
            NWX,NWY=number of windows in X and Y direction, set to 10 by default
            FACTOR=True means that file is result of subtraction of two bias so RMS must be corrected by sqrt(2)
            CF=x.x conversion factor in e/ADU
            NSTD= Number of standard deviation used to discard values (default 6)
            LOG=True  if true => plot in log scale (Y)
            TITLE= used to put a title to the plot
            SAVE= save plot
            BINS= number of bins to make histogram
            RETURN= if True, return a tuple containing mean and median value od std

        """
        # if MASK=True use masked array for computation
        local_copy = self.copy()

        if kargs.get('MASK', True):
            local_copy.mask(**kargs)

        nx = kargs.get('NWX', 10)  # use 10 windows in X by default
        ny = kargs.get('NWY', 10)  # use 10 windows in Y by default
        bins = kargs.get('BINS', 50)  # use 50 bins for histogram by default

        Xi, Xf, Yi, Yf = local_copy.get_windowcoor(*coor)

        # wx = (Xf - Xi)//nx  #NHA /
        # wy = (Yf - Yi)/ny   #NHA /

        # generate array of nx * ny elements
        aux_std = np.zeros((nx, ny))
        aux_mean = np.zeros((nx, ny))
        aux_median = np.zeros((nx, ny))

        windows = subwindowcoor(Xi, Xf, Yi, Yf, **kargs)  # windows is a generator of subwindows
        # for each sub window compute std, mean and median
        for i, j, xi, xf, yi, yf in windows:
            # This should work no matter the image orientation!
            aux_std[i, j] = np.std(local_copy[xi:xf, yi:yf], axis=None)
            aux_mean[i, j] = np.mean(local_copy[xi:xf, yi:yf], axis=None)
            # median doesn't work in masked arrays...
            aux_median[i, j] = np.median(local_copy.get_data()[xi:xf, yi:yf], axis=None)
            '''
            if kargs.get('MASK', True):
                aux_median[i, j] = np.median(self.get_data()[xi:xf, yi:yf], axis=None)
            else:
                aux_median[i, j] = np.median(self[xi:xf, yi:yf], axis=None)
            '''

            if kargs.get('PRINT', False):
                print(
                    f'[{xi: >5d}:{xf: >5d},{yi: >5d}:{yf: >5d}] std:{aux_std[i, j]:.1f} mean:{aux_mean[i, j]:.1f} median:{aux_median[i, j]:.1f}')

        # Compute statistic
        medianval = np.median(aux_median, axis=None)  # median value of boxes
        meanval = np.mean(aux_mean, axis=None)  # mean value of boxes
        stdstd = np.std(aux_std, axis=None)  # stddev of standard deviation of boxes
        medstd = np.median(aux_std, axis=None)  # median value of standard deviation

        # if RETURN is True, return tuple with mean and std
        if kargs.get('RETURN'):
            return (meanval, medianval, medstd, stdstd)

        # TODO print the correct window used for the computation
        print('')
        print(f'Window analysed: [{Xi}:{Xf},{Yi}:{Yf}] devided in {nx*ny} subwindows')
        print(f'MaxVal={local_copy[Xi:Xf, Yi:Yf].max():.2f}  ADUs')
        print(f'MinVal={local_copy[Xi:Xf, Yi:Yf].min():.2f}  ADUs')
        print('')
        print(f'Mean  = {meanval:.2f} +/-{medstd:.3f} ADUs')
        print(f'Median= {medianval:.2f} +/-{medstd:.3f} ADUs')
        print('')
        # change shape of mean array from 2D to 1D
        # TODO  window is not defined....
        im = local_copy[Xi:Xf, Yi:Yf].copy()
        im.shape = (im.shape[0]*im.shape[1], )

        # number of standard deviations to define the mask
        nstd = kargs.get('NSTD', 6)
        # generate a mask with all values inside +/- 5*stddev
        # TODO: Check if this is the best way to produce a mask, or maybe use masked array....
        mask1 = im < (meanval + nstd*medstd)
        mask2 = im > (meanval - nstd*medstd)
        # mask = mask1*mask2
        plt.figure()
        # TODO : Check why histogram is wrong with NSTD less than 6
        if kargs.get('LOG', False):
            plt.hist(im, list(np.linspace((meanval - nstd*medstd),
                                          (meanval + nstd*medstd), bins)), histtype='step', log='True')
        else:
            plt.hist(im, list(np.linspace((meanval - nstd*medstd),
                                          (meanval + nstd*medstd), bins)), histtype='step')

        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('Signals [ADU]')
        if kargs.get('TITLE', ''):
            plt.title('Histogram of pixel values for %s' % kargs.get('TITLE'))
        else:
            plt.title('Histogram of pixel values')
        plt.figtext(0.15, 0.8, 'Mean=%8.2f ADUs' % meanval, fontsize=9,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.figtext(0.15, 0.75, 'Median= %8.2f ADUs' % medianval,
                    fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.figtext(0.15, 0.65, 'Window:[%d:%d,%d:%d]' % (
            Xi, Xf, Yi, Yf), fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))
        if kargs.get('CF'):
            cf = kargs.get('CF')
            if kargs.get('FACTOR'):
                factor = np.sqrt(2.0)
            else:
                factor = 1.0
            cf = cf/factor

            plt.figtext(0.15, 0.7, 'RMS   = %7.3f -e  +/-%7.3f' % (medstd*cf, stdstd*cf),
                        fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))
        else:
            # TODO why I'm dividing by sqrt(2)
            # figtext(0.15,0.7,'RMS   = %7.3f  +/-%7.3f ADUs' % (medstd,stdstd/sqrt(2.0)),fontsize=9,bbox=dict(facecolor='yellow', alpha=0.5))
            plt.figtext(0.15, 0.7, 'RMS   = %7.3f  +/-%7.3f ADUs' %
                        (medstd, stdstd), fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))
        if kargs.get('SAVE'):
            if kargs.get('TITLE', self.filename):
                name = kargs.get('TITLE')
                name = name.replace(' ', '_')
            plt.savefig('Statistic_'+name+'.png')
        if kargs.get('SHOW', False):
            plt.show()

    # TODO: Must be finished
    def extractspect(self, x0, y0, width, angle):
        """
        Extract a tilted spectra like UVES.
        Point1 is the lower left side of the spectra (vertical oriented) or the lower left (horizontally oriented)
        Point2 is the upper left side of the spectra (vertical oriented) or the lower right (horizontally oriented)
        width is the width of spectra

        """
        spectra = []
        # ycoor = []

        # TODO Initialize generator for tilted line
        limits = genspectralimits(self, x0, y0, width, angle)

        for x, yi, yf in limits:
            spectra.append(self.data[x, yi:yf].mean())
            if x == 0:
                y0pos = (yi+yf)/2.0

        return y0pos, np.array(spectra)  # convert list to array

    def avecol(self, *ycoor, **kargs):
        """
        Average and plot the columns data starting from yi till yf
        Syntax:
        im.avecol(yi,yf[,XMIN=xmin,XMAX=xmax,YMIN=ymin,YMAX=ymax,RETURN=True/False][,OVERPLOT=True][,SAVE=True][TITLE='text'])
        XMIN = starting point for plotting the column trace
        XMAX = final point to plot the column trace
        YMAX = maximum Y value to display
        YMIN = minimum Y value to display
        RETURN if True, it will return a vector with the trace values and will not make the plot
        OVERPLOT if True, will not create a new plot and draw over the previous one. NOTE: Doesn't work fine using notebooks
        SAVE, if True will save a png file with the plot
        TITLE will display the 'text' as title of the plot


        """

        y1, y2 = self.get_ycoor(*ycoor)

        if not kargs.get('RETURN', False):
            OVERPLOT = kargs.get('OVERPLOT', False)
            if not OVERPLOT:
                plt.figure()
            plt.grid()
            plottitle = kargs.get('TITLE', '')
            plt.title(plottitle+'\n'+' Col Avrg @ (%d:%d)' % (y1, y2))
            plt.xlabel('Rows')
            plt.ylabel('Signal [ADUs]')
            plt.plot(np.mean(self[self.xi:self.xf, y1:y2], axis=1))

            xi, xf = plt.xlim()
            yi, yf = plt.ylim()

            xl = kargs.get('XMIN', None)
            xm = kargs.get('XMAX', None)
            yl = kargs.get('YMIN', None)
            ym = kargs.get('YMAX', None)
            plt.xlim(xl, xm)
            plt.ylim(yl, ym)
            # figtext(0.15,0.85,'Columns Average @ Y(%d:%d)' % (y1,y2),fontsize=11,bbox=dict(facecolor='yellow', alpha=0.5))

            plt.show()
            if kargs.get('SAVE'):
                dt = datetime.datetime.now()
                dtstr = dt.strftime('_%Y_%m_%d:%H_%M_%S_')
                name = kargs.get('TITLE', self.filename)
                # TODO Remove .fits from self.filename
                name = name.replace('.fits', '')
                name = name.replace(' ', '_')
                plt.savefig('ColAvrg_' + name + dtstr + '.png')

            return None
        return np.mean(self[self.xi:self.xf, y1:y2], axis=1)

    def mediancol(self, *ycoor, **kargs):
        """Average and plot the columns starting from  yi till  yf
        Syntax:
        im.mediancol(yi,yf[,XMIN=xmin,XMAX=xmax,YMIN=ymin,YMAX=ymax,RETURN=True/False][,OVERPLOT=True][,SAVE=True][TITLE='text'])

        XMIN = starting point for plotting the column trace
        XMAX = final point to plot the column trace
        YMAX = maximum Y value to display
        YMIN = minimum Y value to display
        RETURN if True, it will return a vector with the trace values and will not make the plot
        OVERPLOT if True, will not create a new plot and draw over the previous one. NOTE: Doesn't work fine using notebooks
        SAVE, if True will save a png file with the plot
        TITLE will display the 'text' as title of the plot
        """

        y1, y2 = self.get_ycoor(*ycoor)

        if not kargs.get('RETURN', False):
            OVERPLOT = kargs.get('OVERPLOT', False)
            if not OVERPLOT:
                plt.figure()
            plt.grid()
            plottitle = kargs.get('TITLE', '')
            plt.title(plottitle+'\n' + 'Col Median @ (%d:%d)' % (y1, y2))
            # title('Y axis Median')
            plt.xlabel('Rows')
            plt.ylabel('Signal [ADUs]')
            plt.plot(np.median(self[self.xi:self.xf, y1:y2], axis=1))

            xi, xf = plt.xlim()
            yi, yf = plt.ylim()
            xl = kargs.get('XMIN', None)
            xm = kargs.get('XMAX', None)
            yl = kargs.get('YMIN', None)
            ym = kargs.get('YMAX', None)
            plt.xlim(xl, xm)
            plt.ylim(yl, ym)
            # figtext(0.15,0.85,'Y axis Median @ Y(%d:%d)' % (y1,y2),fontsize=11,bbox=dict(facecolor='yellow', alpha=0.5))

            plt.show()
            if kargs.get('SAVE'):
                dt = datetime.datetime.now()
                dtstr = dt.strftime('_%Y_%m_%d:%H_%M_%S_')
                name = kargs.get('TITLE', self.filename)
                # TODO Remove .fits from self.filename
                name = name.replace('.fits', '')
                name = name.replace(' ', '_')
                plt.savefig('ColMedian_' + name + dtstr + '.png')

            return None
        return np.median(self[self.xi:self.xf, y1:y2], axis=1)

    def averow(self, *xcoor, **kargs):
        """
        Syntax: im.averow(xi,xf[,XMIN=xmin,XMAX=xmax,YMIN=ymin,YMAX=ymax,RETURN=True/False][,OVERPLOT=True][,SAVE=True][TITLE='text'])

        Average and plot the row axis data rows starting from xi till xf
        XMIN = starting point for plotting the column trace
        XMAX = final point to plot the column trace
        YMAX = maximum Y value to display
        YMIN = minimum Y value to display
        RETURN if True, it will return a vector with the trace values and will not make the plot
        OVERPLOT if True, will not create a new plot and draw over the previous one. NOTE: Doesn't work fine using notebooks
        SAVE, if True will save a png file with the plot
        TITLE will display the 'text' as title of the plot


        """

        x1, x2 = self.get_xcoor(*xcoor)

        if not kargs.get('RETURN', False):  # instead of plotting, return a vector
            OVERPLOT = kargs.get('OVERPLOT', False)
            if not OVERPLOT:
                plt.figure()
            plt.grid()
            plottitle = kargs.get('TITLE', '')
            plt.title(plottitle+'\n' + ' Row Avrg @ (%d:%d)' % (x1, x2))
            # title('Row Average '+plottitle)
            plt.xlabel('Columns')
            plt.ylabel('Signal [ADUs]')

            plt.plot(np.mean(self[x1:x2, self.yi:self.yf], axis=0))
            # xi, xf = plt.xlim()
            # yi, yf = plt.ylim()
            xl = kargs.get('XMIN', None)
            xm = kargs.get('XMAX', None)
            yl = kargs.get('YMIN', None)
            ym = kargs.get('YMAX', None)
            plt.xlim(xl, xm)
            plt.ylim(yl, ym)
            # figtext(0.15,0.85,'Row Average @ X(%d:%d)' % (x1,x2),fontsize=11,bbox=dict(facecolor='yellow', alpha=0.5))
            plt.show()
            # TODO Add datetime to saved file to make it unique
            if kargs.get('SAVE'):
                dt = datetime.datetime.now()
                dtstr = dt.strftime('_%Y%m%dT%H%M%S')
                name = kargs.get('TITLE', self.filename.replace(' ', '_'))
                '''
                if kargs.get('TITLE'):
                    name = kargs.get('TITLE', self.filename)
                    name = name.replace(' ', '_')
                '''
                plt.savefig(f'RowAvrg_{name}{dtstr}.png')

            return None
        return np.mean(self[x1:x2, self.yi:self.yf], axis=0)

    def medianrow(self, *xcoor, **kargs):
        """
        Syntax: im.medianx(xi,xf[,XMIN=xmin,XMAX=xmax,YMIN=ymin,YMAX=ymax,RETURN=True/False])

        Average and plot the x axis rows starting from row xi till row xf

        XMIN = starting point for plotting the column trace
        XMAX = final point to plot the column trace
        YMAX = maximum Y value to display
        YMIN = minimum Y value to display
        RETURN if True, it will return a vector with the trace values and will not make the plot
        OVERPLOT if True, will not create a new plot and draw over the previous one. NOTE: Doesn't work fine using notebooks
        SAVE, if True will save a png file with the plot
        TITLE will display the 'text' as title of the plot


        """

        x1, x2 = self.get_xcoor(*xcoor)

        if not kargs.get('RETURN', False):
            OVERPLOT = kargs.get('OVERPLOT', False)
            if not OVERPLOT:
                plt.figure()
            plt.grid()
            plottitle = kargs.get('TITLE', '')
            plt.title(plottitle+'\n' + ' Row Median @ (%d:%d)' % (x1, x2))
            # title('X Median')
            plt.xlabel('Columns')
            plt.ylabel('Signal [ADUs]')

            plt.plot(np.median(self[x1:x2, self.yi:self.yf], axis=0))
            # xi, xf = plt.xlim()
            # yi, yf = plt.ylim()
            xl = kargs.get('XMIN', None)
            xm = kargs.get('XMAX', None)
            yl = kargs.get('YMIN', None)
            ym = kargs.get('YMAX', None)
            plt.xlim(xl, xm)
            plt.ylim(yl, ym)
            # figtext(0.15,0.85,'X Median @ X(%d:%d)' % (x1,x2),fontsize=11,bbox=dict(facecolor='yellow', alpha=0.5))

            plt.show()
            if kargs.get('SAVE'):
                dt = datetime.datetime.now()
                dtstr = dt.strftime('_%Y_%m_%d:%H_%M_%S_')
                if kargs.get('TITLE'):
                    name = kargs.get('TITLE', self.filename)
                    name = name.replace(' ', '_')
                plt.savefig('RowMedian_' + name + dtstr + '.png')

            return None
        return np.median(self[x1:x2, self.yi:self.yf], axis=0)

    def set_value(self, value=1000, *coor, **kargs):
        """
        Initialize all the pixels to the given value (from 0 up to 65535)
        If STD is not zero, it also adds gaussian noise with amplitud STD

        """

        Xi, Xf, Yi, Yf = self.get_windowcoor(*coor)  # get area coordinates from coor
        self.data[Xi:Xf, Yi:Yf] = value  # fill numpy array with value

        stddev = kargs.get('STD', 0)
        if stddev != 0:
            noise = np.random.standard_normal([Xf-Xi, Yf-Yi])
            self.data[Xi:Xf, Yi:Yf] += noise*stddev

    def get_data(self):
        """
        Returns the whole numpy array which contains the image
        """
        # return self.data
        return np.copy(self.data)

    def get_xi(self):
        """

        """
        return self.xi

    def get_xf(self):
        """

        """
        return self.xf

    def get_yi(self):
        """

        """
        return self.yi

    def get_yf(self):
        """

        """
        return self.yf

    def set_xps(self, xps):
        """
        Set the width of presacn region on x
        """
        self.xps = xps

    def get_xps(self):
        """
        Read out the width of prescan region in x
        """
        return self.xps

    def set_xos(self, xos):
        """
        Set the width of oversacn region on x
        """
        self.xos = xos

    def get_xos(self):
        """
        Read out the width of overscan region in x
        """
        return self.xos

#
    def set_yps(self, yps):
        """
        Set the width of presacn region on y
        """
        self.yps = yps

    def get_yps(self):
        """
        Read out the width of prescan region in y
        """
        return self.yps

    def set_yos(self, yos):
        """
        Set the width of oversacn region on y
        """
        self.yos = yos

    def get_yos(self):
        """
        Read out the width of overscan region in y
        """
        return self.yos

    def format(self):
        """
        Return a tuple with the X and Y dimension of the image

        ex:
        b1=Image('B1.fits')
        b1.format()
        """
        return self.shape

    def subwindows(self, nx=10, ny=10):
        """
        Defines how many subwindows are used for the analysis,
        nx is the number of subwindows in the x direction
        ny is the number of subwindows in the y direction
        """
        self.nx = nx
        self.ny = ny
        print(f'Xsubwindows={self.nx}, Ysubwindows={self.ny}')

    def fftcol(self, ReadOutSpeed=100, Vdelay=5, ColStart=None, ColEnd=None, **kargs):
        """

        ReadOutSpeed: speed in kps used to read the image
        Vdelay: time in pixel to make a parallel transfer
        ColStart: first column to analize
        ColEnd:final column to analize

        Compute the fft for each column in the image, then add all of them and devide by
        the number of columns. This reduce the noise and allows to clearly detect the
        peaks. If the function is called with the option PLOT=False, then no plotting
        is produce but the function returns a vector containing the Power Spectral Density
        that can be plotted later
            ex:
            b1=Image('B1.fits')
            b2=Image('B2.fits')
            freq=b1.fftcolfreq()
            psd1=b1.fftcol(PLOT=False)
            psd2=b2.fftcol(PLOT=False)
            plot(freq,psd1)
            plot(freq,psd2)

        kargs options:
        DEBIAS: True/False => remove mean level from signal (default=True)
        PLOT: True/False => make a plot of the fft (default=True)
        RETURN: True/False => return an array with the values of fft (default=False)
        BSTART: int => starting bin in the plot, default=2 (bin 0 has the DC component)
        NOTE: string => add note to plot
        SAVE: True/False  => save plot

        freq=b1.fftcol(RETURN=true)
        b1.fftcol(625, 5, BSTAT=3) make fftcol with readout speed of 625K, Vertical delay = 5 pixel times
         and start the plot from bin 3






        DONE: copy self.data to other array and remove mean level, if DEBIAS=True
        """
        if kargs.get('DEBIAS', True):
            data = self.data
            data = data - data.mean()
        else:
            data = self.data

        if ColStart == None:
            ColStart = 0
        if ColEnd == None:
            ColEnd = self.shape[1]

        # compute frequency
        Tp2p = (1.0/ReadOutSpeed)/1000.0

        # Time between 2 vertical pixels is equal to time from pix to pixel
        # multiplied by number of horizontal pixels plus the vertical delay for the parallel transfer
        Ts = Tp2p*(self.shape[1] + Vdelay)
        print(f'File: {self.filenameinfo(RETURN=True)}')
        if kargs.get('PLOT', True):
            print(f'Time between vertical pixels: {Ts} s')
        # NPix is number of pixels in the column
        NPix = self.shape[0]
        ColS = ColStart
        ColE = ColEnd

        if kargs.get('PLOT', True):
            print(f'First and Last column to analyse: {ColS} {ColE}')

        DimFFT = 2**(nextpow2(NPix))  # find closer power of 2
        FirstPix = 0
        LastPix = DimFFT
        Fs = 1/Ts
        if kargs.get('RETURN', True):
            print(f'Number of original pixels: {NPix}')
            print(f'Dimension FFT: {DimFFT} pix')

        # compute freq vector for ploting
        freq = (Fs/2) * np.linspace(0, 1, num=DimFFT//2)

        if kargs.get('RETURN', True):
            print(f'Largo freq={len(freq)}')
            print('')
            print(f'Each freq bin is equal to: {Fs/DimFFT} Hz')
            print(f'Maximum freq : {Fs/2}')

        # prepare the hanning window
        hwindow = np.hanning(DimFFT)
        if kargs.get('PLOT', True):
            print(f'Largo hwindow={len(hwindow)}')

        Acum = np.zeros(DimFFT//2)

        for x in range(ColS, ColE):
            # col = self.data[FirstPix:LastPix,x]
            col = data[FirstPix:LastPix, x]
            col = col * hwindow  # apply hanning window
            TransF = np.fft.fft(col)/DimFFT  # NPix  #TODO check if this is correct
            AbsTransF = 2 * abs(TransF[:DimFFT//2])
            Acum = Acum + AbsTransF

        Acum = Acum/(ColE-ColS)  # take the average of the ffts

        if kargs.get('RETURN', False):
            return freq, Acum
        if kargs.get('PLOT', True):
            plt.figure()
            binstart = kargs.get('BSTART', 2)
            plt.grid()
            # skip the DC component in the plot to improve autoscale
            plt.plot(freq[binstart:], Acum[binstart:])

            note = kargs.get('NOTE', None)
            if note:
                plt.figtext(0.15, 0.85, '%s' % note, fontsize=11,
                            bbox=dict(facecolor='yellow', alpha=0.5))

            plt.xlabel('Freq [Hz] with resolution='+str(Fs/DimFFT)+' Hz')
            plt.ylabel('Intensity')
            plt.grid(True)
            plt.title('Low freq FFT analysis (Vertical transfer direction)')
            plt.title('Average FFT on columns '+kargs.get('TITLE', ''))
            if kargs.get('SAVE', False):
                plt.savefig('fftcol_%s_CCDnr%d.png' % (self.filenameinfo(RETURN=True), self.ext))
            plt.show()

    def fftrow(self, ReadOutSpeed=100, RowStart=None, RowEnd=None, **kargs):
        """

        ReadOutSpeed: speed in kps used to read the image
        RowStart: first row to analize
        RowEnd:final row to analize


        Compute the fft for each row in the image, then add all of them and devide by the
        number of rows. This reduce the noise and allows to clearly detect the peaks. If the
        function is called with the option PLOT=False, then no plotting is produce but the
        function returns a vector containing the Power Spectral Density that can be plotted
        later
        ex:
        b1=Image('B1.fits')
        b2=Image('B2.fits')
        freq=b1.fftrowfreq()
        psd1=b1.fftrow(PLOT=False)
        psd2=b2.fftrow(PLOT=False)
        plot(freq,psd1)
        plot(freq,psd2)

        kargs options:
        DEBIAS: True/False => remove mean level from signal (default=True)
        PLOT: True/False => make a plot of the fft (default=True)
        RETURN: True/False => return an array with the values of fft (default=False)
        BSTART: int => starting bin in the plot, default=2 (bin 0 has the DC component)
        NOTE: string => add note to plot
        SAVE: True/False  => save plot

        freq=b1.fftcol(RETURN=true)
        b1.fftrow(625, 5, BSTAT=3) make fftrow with readout speed of 625K, Vertical delay = 5 pixel times
         and start the plot from bin 3
        """

        # by default, remove bias level
        if kargs.get('DEBIAS', True):
            data = self.data
            data = data - data.mean()
        else:
            data = self.data

        if RowStart == None:
            RowStart = 0
        if RowEnd == None:
            RowEnd = self.shape[0]

        # pixel time
        Ts = 1.0/ReadOutSpeed
        Ts = Ts/1000.0

        print(f'File: {self.filenameinfo(RETURN=True)}')

        if kargs.get('PLOT', True):
            print(f'Time between horizontal pixels: {Ts}')
        # number of pixels
        NPix = self.shape[1]
        RowS = RowStart
        RowE = RowEnd

        if kargs.get('PLOT', True):
            print(f'First and Last row to analyse: {RowS} {RowE}')
        # dimension of FFT
        DimFFT = 2**(nextpow2(NPix))
        FirstPix = 0
        LastPix = DimFFT
        if kargs.get('PLOT', True):
            print(f'Number of original pixels: {NPix}')
            print(f'Dimension FFT: {DimFFT} pix')

        # compute maximum freq
        Fs = 1/Ts

        # compute freq vector for ploting
        freq = (Fs/2)*np.linspace(0, 1, num=DimFFT//2)

        if kargs.get('PLOT', True):
            print(f'Length freq={freq}')
            print('')
            print(f'Each freq bin is equal to: {Fs/DimFFT} Hz')
            print(f'Maximum freq : {Fs/2}')

        # prepare the hanning window
        hwindow = np.hanning(DimFFT)
        if kargs.get('RETURN', True):
            print(f'Length hwindow={len(hwindow)}')

        Acum = np.zeros(DimFFT//2)

        for y in range(RowS, RowE):
            # row = self.data[y,FirstPix:LastPix]
            row = data[y, FirstPix:LastPix]
            row = row*hwindow
            TransF = np.fft.fft(row)/DimFFT  # NPix
            AbsTransF = 2*abs(TransF[:DimFFT//2])
            Acum = Acum+AbsTransF

        Acum = Acum/(RowE-RowS)

        if kargs.get('RETURN', False):
            return freq, Acum

        if kargs.get('PLOT', True):
            # if option.get('PLOT',True):
            plt.figure()
            binstart = kargs.get('BSTART', 2)
            plt.grid()
            # skip the DC component in the plot to improve autoscale
            plt.plot(freq[binstart:], Acum[binstart:])

            note = kargs.get('NOTE', None)
            if note:
                plt.figtext(0.15, 0.85, '%s' % note, fontsize=11,
                            bbox=dict(facecolor='yellow', alpha=0.5))

            plt.xlabel('Freq [Hz] with resolution='+str(Fs/DimFFT)+' Hz')
            plt.ylabel('Intensity')
            plt.grid(True)
            plt.title('Average FFT on rows '+kargs.get('TITLE', ''))
            if kargs.get('SAVE', False):
                plt.savefig('fftrow_%s_CCDnr%d.png' % (self.filenameinfo(RETURN=True), self.ext))
            plt.show()

    def fft2D(self, **kargs):
        """
        Perform the discrete fourier transform of an image.
        First it crop the image to the closest power of 2

        kargs:
        NSTD (default=1) number of stddev used to define lower and upper cut
        levels

        """

        # remove mean value
        image = self.data - self.data.mean()
        Dim = 2**nextpow2(min(image.shape))
        # now crop image to this Dim
        image = image[:Dim, :Dim]
        freq = np.abs(np.fft.fft2(image))
        freqshifted = np.fft.fftshift(freq)
        logfreq = np.log(freqshifted)
        # plt.hist(logfreq.ravel(),bins=100)
        vmin = logfreq.mean() - kargs.get('NSTD', 1)*logfreq.std()
        vmax = logfreq.mean() + kargs.get('NSTD', 1)*logfreq.std()
        plt.imshow(logfreq, vmin=vmin, vmax=vmax, interpolation="none", cmap='gray')
        return logfreq

    def bitcounts(self, *coor):
        """
        Count the bit frequency
        Might be useful to detect missing bits in the ADC
        """
        labels = ['$2^0$', '$2^1$', '$2^2$', '$2^3$', '$2^4$', '$2^5$', '$2^6$', '$2^7$', '$2^8$',
                  '$2^9$', '$2^{10}$', '$2^{11}$', '$2^{12}$', '$2^{13}$', '$2^{14}$', '$2^{15}$', '$2^{16}$']

        Xi, Xf, Yi, Yf = self.get_windowcoor(*coor)

        # convert CCD data to 16 bit integers
        aux = self.data[Xi:Xf, Yi:Yf].astype(np.uint16)

        # print(aux.dtype)
        totalpix = (Xf-Xi)*(Yf-Yi)
        # print(type(totalpix))
        bitfreq_0 = []
        bitfreq_1 = []
        print(
            f'Mean value= {np.mean(self.data[Xi:Xf, Yi:Yf])},  StdDev= {np.std(self.data[Xi:Xf, Yi:Yf])}')
        print(f'Number of pixels: {totalpix}')
        for i in range(16):
            bitp = int(2**i)
            maux = np.ma.masked_where(aux & bitp, aux, copy=False)
            nbits0 = maux.count()
            bitfreq_0.append(nbits0)
            nbits1 = totalpix - nbits0
            bitfreq_1.append(nbits1)
            print(f'Bit value :{bitp:5d} @0: {nbits0:7d}  @1: {nbits1:7d}')

        xlocations = np.array((list(range(len(bitfreq_0)))))
        width = 0.7
        # fig = plt.figure()
        p1 = plt.bar(xlocations, bitfreq_0, width, color='c')
        p2 = plt.bar(xlocations, bitfreq_1, width, color='b', bottom=bitfreq_0)
        plt.xticks(xlocations+width/2.0, labels)
        plt.xlim(0, xlocations[-1]+width*2)
        plt.legend((p1[0], p2[0]), ('0', '1'))
        plt.title('Bit frequency analysis')
        plt.ylabel('Number of counts')
        plt.axhline(y=totalpix/2, color='r')
        # uncomment following line if you want text in x axis rotated 45degree
        # fig.autofmt_xdate()
        plt.show()

    def addnormalnoise(self, stddev):
        """
        Add normal distribution noise with amplitud stddev to the image

        b1.addnormalnoise(10.0)

        """
        noise = np.random.standard_normal([self.shape[0], self.shape[1]])

        self.data += noise*stddev

    def addsinusoidalnoise(self, fnvector, ampvector, rofreq, vdelay, **kargs):
        """
        Add sinusoidal noise of frequencies contained in an array.
        fnvector: vector with noise frequencies to inject in Hertz
        ampvector: vector with amplitud for the respective frequency in ADUs
        rofreq: readout frequency of ccd serial register (ex: 100 means 100kps)
        vdelay: vertical delay in pixel time to transfer one line towards the serial register

        example
        b1.addsinusoidalnoise([10,20,30],[1,2,3], 100, 5)
        => add sinusoids of 10Hz, 20Hz and 30Hz with amplitudes of 1, 2 and 3 ADUs


        """

        jit = kargs.get('JITTER', 0)
        jitter = jit * (10**-9)

        pixeltime = 1.0/(rofreq*1000)
        shotnoise = np.random.standard_normal(
            [self.shape[0], self.shape[1]])  # array to store shot noise
        aux = np.zeros((self.shape[0], self.shape[1]))
        signal = np.zeros((self.shape[0], self.shape[1]))

        # auxserial = zeros((self.shape[1],))

        col = np.arange(0, self.shape[1])
        row = np.arange(0, self.shape[0])
        row = row.reshape((self.shape[0], 1))*self.shape[1] + vdelay
        row[0, 0] = 0
        data = col + row

        if jitter == 0:
            jitter_matrix = np.ones((self.shape[0]*self.shape[1]))
        else:
            jitter_matrix = np.random.normal(1.0, jitter, self.shape[0]*self.shape[1])
        jitter_matrix = jitter_matrix.reshape((self.shape[0], self.shape[1]))
        jitter_matrix = jitter_matrix * pixeltime

        aux = data * jitter_matrix

        # generate on vector containing pixel time for serial register
        # for i in range(len(auxserial)):
        #    auxserial[i] = i*pixeltime

        # generate full array with pixel time for all pixels in de image
        # for i in range(self.shape[0]):
        #    aux[i,:] = aux[i,:]+(auxserial[:]+(i*auxserial[-1]+pixeltime*vdelay))

        # add the sinusoids now
        for i in range(len(fnvector)):
            signal = signal + ampvector[i]*np.sin(2*np.pi*fnvector[i]*aux)

        # compute shot noise
        sqrt_signal = np.sqrt(abs(signal))
        # add it to signal
        signal = signal+shotnoise*sqrt_signal

        # now add generated signal to ccd pixels
        self.data = self.data+signal
