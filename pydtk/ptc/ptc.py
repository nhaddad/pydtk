#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:28:54 2018

@author: nhaddad
"""
import numpy as np
from numpy import ma
from pydtk import Image
from scipy import stats
import matplotlib.pyplot as plt
from pydtk.utils.utilsfunc import subwindowcoor
from pydtk.utils.utils import medianstack
from pydtk.utils.utils import meanstack


def gain(imagelist, *coor, **kargs):
    """
    Compute the gain of the system using 2 Bias and 2 FF. The procedure devides the window
    in NWX*NWY subwindows and computes the Gain for each one of them and then computes the mean
    value and display an histogram. If the windows coordinates are not given, it will use the full
    CCD.

    Syntax:
    gain(imagelist[,xi,xf,yi,yf][,NWX=10][,NWY=10][,VERBOSE=True/False][,SAVE=True][,TITLE='Graph Title'][,RETURN=True/False][, MEDIAN=True/False])

    Note: the image list must contain 2 bias and 2 ff in that order!
    imagelist can be a list of names o list of images
    b1,b2= bias images
    f1,f2= ff images
    *coor=[xi,xf,yi,yf] = coordinates of the window to analize (should be a flat region)
    kargs
    -------
    VERBOSE=True => print intermediate values
    TITLE='Graph Title' => put a title in the graph
    SAVE=True/False => if True, it saves the graph in pnp format
    RETURN=True/False => if True, return only ConFAc without plots
    MEDIAN=True/False => default True, computes median instead of mean
    RATIO=True/FALSE => defaul True. just change the way the FPN is elliminated. Both
    methods give almost the same rusults
    NWX= number of windows in X direction (default 10)
    NWY= number of windows in Y direction (default 10)
    EXT=image extension to load, default 0


    """
    ext = kargs.get('EXT', 0)

    if len(imagelist) != 4:
        print('imagelist len different from 4')
        exit

    # check if imagelist are images or filenames
    if all([isinstance(i, str) for i in imagelist]):
        images = [Image(i, ext) for i in imagelist]
        b1 = images[0]
        b2 = images[1]
        ff1 = images[2]
        ff2 = images[3]
    else:
        b1 = imagelist[0]
        b2 = imagelist[1]
        ff1 = imagelist[2]
        ff2 = imagelist[3]

    nwx = kargs.get('NWX', 10)  # set number of windows in x direction
    nwy = kargs.get('NWY', 10)  # set number of windows in y direction

    x1, x2, y1, y2 = b1.get_windowcoor(*coor)
    # print(x1,x2,y1,y2)

    # now work with cropped images, where the signal is more or less flat....
    b1 = b1.crop(x1, x2, y1, y2)
    b2 = b2.crop(x1, x2, y1, y2)
    ff1 = ff1.crop(x1, x2, y1, y2)
    ff2 = ff2.crop(x1, x2, y1, y2)

    dbiasff1 = ff1 - b1  # debiased FF1
    dbiasff2 = ff2 - b2  # debiased FF2
    meanff2 = dbiasff2.mean()
    meanff1 = dbiasff1.mean()
    ratio = meanff1/meanff2
    # print(ratio)

    if kargs.get('VERBOSE', False):
        print("format images X=%d pix Y=%d pix" % b1.format())
        print(("Nx:%d Ny:%d X1:%d X2:%d Y1:%d Y2:%d WX:%d WY:%d") %
              (nwx, nwy, x1, x2, y1, y2, (x2-x1)//nwx, (y2-y1)//nwy))  # NHA /
        print("")
        print("meanff2 =%f" % meanff2)

    dbiasff2 = dbiasff2*ratio
    dbias_ff_diff = dbiasff1 - dbiasff2
    dbias_ff_sig = (dbiasff1 + dbiasff2)/2.0

    # compute difference of 2 bias to get the RON
    dbias = b1 - b2

    # generate auxiliary arrays of nwx * nwy elements and initialize to zero
    meansig = np.zeros((nwx, nwy))
    stdff = np.zeros((nwx, nwy))
    stdbias = np.zeros((nwx, nwy))
    cf = np.zeros((nwx, nwy))
    signal = (dbiasff1 / dbiasff2)
    stdsig = np.zeros((nwx, nwy))

    # windows is a generator of subwindows
    windows = subwindowcoor(0, b1.shape[0], 0, b1.shape[1], **kargs)
    for i, j, xi, xf, yi, yf in windows:
        # compute mean value on each window for normalized ff
        meansig[i, j] = np.mean(dbias_ff_sig[xi:xf, yi:yf])
        # compute standard deviation on each window for normalized ff
        stdsig[i, j] = np.std(dbias_ff_diff[xi:xf, yi:yf])/np.sqrt(2.0)
        cf[i, j] = meansig[i, j] / (stdsig[i, j]**2)  # compute CF for each window
        # compute standard deviation for each window of bias difference
        stdbias[i, j] = np.std(dbias[xi:xf, yi:yf])/np.sqrt(2.0)

        if kargs.get('VERBOSE', False):
            print(("X(%d,%d) Y(%d,%d) Mean:%7.2f stdff:%7.3f  CF:%5.3f") %
                  (xi+x1, xf+x1, yi+y1, yf+y2, meansig[i, j], stdsig[i, j], cf[i, j]))

    if kargs.get('MEDIAN', True):
        ConFac = np.median(cf, axis=None)
        RON = np.median(stdbias, axis=None)

    else:
        ConFac = np.mean(cf, axis=None)
        RON = np.mean(stdbias, axis=None)  # RON in ADUs

    # RON =  RMS / sqrt(2)    #RON in ADUs
    RONe = RON * ConFac     # RON in electrons

    CFstd = np.std(cf, axis=None)/np.sqrt(nwx*nwy)

    # Check if run as ROUTINE, in that case return only the Conversion Factor and don't continue with plotting
    if kargs.get('RETURN', False):
        return x1, x2, y1, y2, ConFac, RONe, meanff2
    else:
        plt.figure()

    print("*******************************************")
    print(("*CF  =%5.3f +/-%5.3f e/ADU") % (ConFac, CFstd))
    print(("*RON =%6.3f -e") % (RONe))
    print(("*RON =%6.3f ADUs") % RON)
    print("*******************************************")

    # change shape of cf array to later compute the standard deviation and also make the histogram

    cf.shape = (nwx*nwy, )
    cfstd = np.std(cf, axis=None)
    plt.clf()
    plt.hist(cf, range=(ConFac - 3*cfstd, ConFac + 3*cfstd), bins=20)
    plt.figtext(0.15, 0.8, ("CF mean=%5.3f +/-%5.3f e/ADU") %
                (ConFac, CFstd), fontsize=11, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.figtext(0.15, 0.75, ("RON =%6.3f -e") %
                (RONe), fontsize=11, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.figtext(0.15, 0.70, ("Computed @ %6.3f ADUs") % (np.mean(meansig)),
                fontsize=11, bbox=dict(facecolor='yellow', alpha=0.5))

    Title = kargs.get('TITLE', '')
    plt.title(Title)
    filetitle = Title.replace(' ', '_')
    plt.show()

    if kargs.get('SAVE', False):
        plt.savefig('ConFac_'+filetitle+'.png')


def linearity_residual(imagelist, *coor, **kargs):
    """
    Compute linearity residual using an image list starting
    with 2 bias and then pairs of FF at diferent levels
    LR = 100*(1 -(Sm/Tm)/(S/t))

    TODO: need to complete!!
    """
    MAXSIGNAL = kargs.get('MAXSIGNAL', 65535.0)
    VERBOSE = kargs.get('VERBOSE', False)

    # read coordinates of first image
    x1, x2, y1, y2 = Image(imagelist[0], ext).get_windowcoor(*coor)


def ptc_ffpairs(imagelist, *coor, **kargs):
    """
    TODO: Need to be finished !!
    NHA

    Perform ptc plot for pairs of ff at same level.
    The pairs of ff should have the same light level.
    The first 2 images in the list must be bias
    To eliminate the FPN, the 'shotnoise' image is computed as the subtraction
    of two debiased flat field images
    optional kargs arguments:
    FACTOR (default = 2.0)
    MAXSIGNAL (default 65535)  => compute PTC only for signal values less than MAXSIGNAL
    VERBOSE (default=False)  ==> print out table with signal and variance

    """

    order = kargs.get('ORDER', 1)  # order of polynomial regression
    if order > 2:
        order = 2

    MAXSIGNAL = kargs.get('MAXSIGNAL', 65535.0)
    VERBOSE = kargs.get('VERBOSE', False)
    ext = kargs.get('EXT', 0)

    # read coordinates of first image
    x1, x2, y1, y2 = Image(imagelist[0], ext).get_windowcoor(*coor)

    oddimageindex = list(range(3, len(imagelist), 2))
    evenimageindex = list(range(2, len(imagelist), 2))

    # Read in bias1 and bias2
    bias1 = Image(imagelist[0], ext).crop(x1, x2, y1, y2)
    bias2 = Image(imagelist[1], ext).crop(x1, x2, y1, y2)

    bias_dif = bias2 - bias1
    # mask out all pixels with value greater or lower than  3*std
    bias_dif.mask()

    # Separate images in even and odd (crop the images..)
    ff1 = [Image(imagelist[i], ext).crop(x1, x2, y1, y2) for i in oddimageindex]
    ff2 = [Image(imagelist[i], ext).crop(x1, x2, y1, y2) for i in evenimageindex]

    # remove bias from both ff images
    ff1d = [(image - bias1) for image in ff1]
    ff2d = [(image - bias2) for image in ff2]

    if kargs.get('USE_FFMEAN', False):
        ffmean = [(image1/image2)*image2.mean() for image1, image2 in zip(ff1d, ff2d)]
    else:
        ffmean = [(image1+image2)/2.0 for image1, image2 in zip(ff1d, ff2d)]

    shotnoise = [(image1 - image2) for image1, image2 in zip(ff1d, ff2d)]

    signal = [image.mean() for image in ff1d]  # ffmean]
    variance = [image.var()/2.0 for image in shotnoise]

    # Need to sort both signal and variance according to list containing mean signal
    zipped = zip(signal, variance)
    zipped_sorted = sorted(zipped)

    # remove signal,variance pairs where signal is above MAXSIGNAL
    zipped_sorted = [x for x in zipped_sorted if x[0] <= MAXSIGNAL]

    # Now we unpack to get back signal and variance sorted
    signal, variance = zip(*zipped_sorted)

    if kargs.get('VERBOSE', False):
        print('Mean signal    Variance')
        for s, v in zip(signal, variance):
            print(' {:6.1f}   {:6.1f}'.format(s, v))

    # compute polynomial coeficients
    coefts = np.polyfit(signal, variance, order)
    polyts = np.poly1d(coefts)
    # compute the fitted values for variance
    variance_fitted = np.polyval(polyts, signal)

    # print('Intercept = {}'.format(polyts(0)))

    fig = plt.figure()  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

    ax.set_ylabel('Variance')
    ax.set_xlabel('Signal')
    ax.grid(True)
    ax.set_title('Photon Transfer Curve')

    # plot variance v/s signal
    # figure()
    ax.plot(signal, variance, 'b.')
    ax.plot(signal, variance_fitted, 'r-')

    cf = 1/coefts[0]

    if order == 1:
        cf = 1/coefts[0]
        print('Extension: {}   CF = {:2.3f} -e/ADU   RON = {:2.3f} -e'.format(ext,
                                                                              cf, cf * bias_dif.std()/np.sqrt(2.0)))
    elif order == 2:
        cf = 1/coefts[1]
        print('Extension: {}   CF = {:2.3f} -e/ADU   RON = {:2.3f} -e'.format(ext,
                                                                              cf, cf * bias_dif.std()/np.sqrt(2.0)))


def ron_adu(b1, b2, *coor, **kargs):
    """
    Take two bias images, subtract one from another and
    then compute the std for many windows. Compute the median of the RMS

    Syntax:
    ron_adu(b1, b2, 200,400, 300, 600)
    ron_adu(b1, b2, 200,400, 300, 600, PRINT=True)  Print individual values
    ron_adu(b1, b2, 200,400, 300, 600, HIST=True)   Plot histogram

    TODO: return mean instead of median
    DONE: make histogram plot
    DONE: provide RETURN option

    """

    nwx = kargs.get('NWX', 10)  # set number of windows in x direction
    nwy = kargs.get('NWY', 10)  # set number of windows in y direction

    x1, x2, y1, y2 = b1.get_windowcoor(*coor)

    # compute difference of images
    biasdiff = b1-b2
    # prepare array to receive std values
    std_biasdiff = np.zeros((nwx, nwy))

    # TODO modify to use subwindows to generate them
    # compute stddev and mean for every subwindow
    windows = subwindowcoor(x1, x2, y1, y2, **kargs)
    for i, j, xi, xf, yi, yf in windows:
        std_biasdiff[i, j] = biasdiff[xi:xf, yi:yf].std()/np.sqrt(2)
        if kargs.get('PRINT', False):
            print(f'[{xi:5}:{xf:5},{yi:5}:{yf:5}] => {std_biasdiff[i, j]:.3}')

    diff_median = np.median(std_biasdiff, axis=None)

    if kargs.get('RETURN', False):
        return np.median(std_biasdiff, axis=None)  # /np.sqrt(2.0)
    else:
        print(f'RON @ [{x1}:{x2},{y1}:{y2}] = {np.median(std_biasdiff, axis=None):2.2} ADUs')
    if kargs.get('HIST', False):
        std_biasdiff.shape = (nwx*nwy, )
        diff_std = np.std(std_biasdiff, axis=None)
        print(f'StdDev = {diff_std:.2}')
        plt.clf()
        plt.hist(std_biasdiff, range=(diff_median - 3*diff_std, diff_median + 3*diff_std), bins=30)
        plt.show()


def correctTDIShift(bias, tdi1, tdi2):
    image1 = tdi1 - bias
    image2 = tdi2 - bias
    line1 = image1.avecol(int(image1.shape[1]/2-10), int(image1.shape[1]/2+10), RETURN=True)
    line2 = image2.avecol(int(image1.shape[1]/2-10), int(image2.shape[1]/2+10), RETURN=True)
    rms12 = []
    rms21 = []

    for i in range(100):
        dif12 = line1[i: 1000+i] - line2[:1000]
        dif21 = line2[i: 1000+i] - line1[:1000]
        sqr12 = dif12 * dif12
        sqr21 = dif21 * dif21
        rms12.append(np.sqrt(sqr12.sum()))
        rms21.append(np.sqrt(sqr21.sum()))
    arms12 = np.array(rms12)
    arms21 = np.array(rms21)
    if np.argmin(arms12) == 0 and np.argmin(arms21) == 0:
        return image1, image2
    elif np.argmin(arms12) == 0 and np.argmin(arms21) > 0:
        return image1.crop(0, image1.shape[0]-np.argmin(arms21), 0, image1.shape[1]),\
            image2.crop(np.argmin(arms21), image2.shape[0], 0, image2.shape[1])
    elif np.argmin(arms12) > 0 and np.argmin(arms21) == 0:
        return image1.crop(np.argmin(arms12), image1.shape[0], 0, image1.shape[1]),\
            image2.crop(0, image2.shape[0]-np.argmin(arms12), 0, image2.shape[1])


def ptc_shutterless(bias, tdi, *coor, **kargs):
    """
    The shutterless image is obtained defocusing an spot and then reading out
    the CCD while the shutter is open
    ptc_shutterless(bias, tdi, xi, xf, yi, yf, AXIS=1, ORDER=2, NSTD=3)
    Use one bias image and a defocused spot which generate a trail over the CCD
    as the shutter is kept open while reading.
    AXIS = 1 => the readout is done along the ROWS
    AXIS = 0 => the readout is done along the columns
    ORDER= 2 or 1, is the order of polynomia to fit the PTC curve
    NSTD=3 => mask out pixels which are more than 3 std from the mean
    This method can be used on FORS2
    """
    nstd = kargs.get('NSTD', 3)  # factor to elliminate outlayers
    order = kargs.get('ORDER', 2)
    axis = kargs.get('AXIS', 1)
    maxsignal = kargs.get('MAXSIGNAL', 65000)

    tdidb = tdi - bias

    x1, x2, y1, y2 = bias.get_windowcoor(*coor)

    tdidbc = tdidb.crop(x1, x2, y1, y2)

    datos = tdidbc.get_data()
    zsco = stats.zscore(datos, axis=axis, ddof=1)
    maskzscore = np.ma.masked_outside(zsco, -1*nstd, nstd)
    maskmax = np.ma.masked_greater(datos, maxsignal)

    datosz = np.ma.masked_where(np.ma.getmask(maskzscore), datos)
    datoszm = np.ma.masked_where(datosz >= maxsignal, datosz)
    tdidbc.data = datoszm

    signal = tdidbc.mean(AXIS=axis)
    signalvar = tdidbc.var(AXIS=axis)
    cf = signal/signalvar

    coefts = np.polyfit(signal[signal <= (maxsignal*0.9)],
                        signalvar[signal <= (maxsignal*0.9)], order)
    polycoef = np.poly1d(coefts)
    var_fitted = np.polyval(polycoef, signal[:])

    # if RETURN equal True, return signal_masked, variance masked, fitted variance and CF
    if kargs.get('RETURN', False):
        if order == 1:
            return signal, signalvar, var_fitted, (1/coefts[0])
        else:
            return signal, signalvar, var_fitted, (1/coefts[1])

    else:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10, 5))
        # plot variance vs signal without masking yet
        ax1.plot(signal, cf, '.b')
        ax1.grid()
        ax1.set_title('Photon Transfer Curve')
        ax1.set_ylabel('Variance [ADU]**2')
        ax1.set_xlabel('Signal [ADU]')

        # plot variance vs signal WITH masking
        ax2.plot(signal, signalvar, '.b', signal, var_fitted, 'r')
        ax2.grid()
        title = 'PTC  CF=%f'
        if order == 2:
            title = title % (1/coefts[1])
        else:
            title = title % (1/coefts[0])
        ax2.set_title(title)
        ax2.set_ylabel('Variance [ADU]**2')
        ax2.set_xlabel('Signal [ADU]')


def ptc_2tdi(bias, tdi1, tdi2, *coor, **kargs):
    """
    Perform ptc plot with one bias and 2 tdi images.
    ex:
    ptc_2tdi(b1,tdi1,tdi2,50, 2000, 100, 170)
    compute CF using 2 bias and 2 FF in an area defined by [50:2000,100:170] plot the ptc curve and compute the CF using
    a first order polynomia

    ptc_2tdi(b1,tdi1,tdi2,50, 2000, 100, 170, RETURN=True)
    compute CF using 2 bias and 2 FF in an area defined by [50:2000,100:170] return the vectors and the CF, using
    a first order polynomia

    ptc_2tdi(b1,tdi1,tdi2,50, 2000, 100, 170, ORDER=2)
    compute CF using 2 bias and 2 FF in an area defined by [50:2000,100:170] plot the ptc curve, and use a
    polynomia of order 2


    The 2 tdi images should have a slope in flux to compute the ptc.
    To eliminate the FPN, the 'shotnoise' image is computed as the subtraction
    of two debiased flat field images
    optional kargs arguments:
    NSTD (default = 3) Default number of std deviation to elliminate outlayers
    VERBOSE (default=False)

    This method can be used on the TestBench
    """

    nstd = kargs.get('NSTD', 3)  # factor to elliminate outlayers
    order = kargs.get('ORDER', 2)
    axis = kargs.get('AXIS', 1)  # columns

    dbiasff1, dbiasff2 = correctTDIShift(bias, tdi1, tdi2)

    x1, x2, y1, y2 = dbiasff1.get_windowcoor(*coor)

    dbiasff1 = dbiasff1.crop(x1, x2, y1, y2)
    dbiasff2 = dbiasff2.crop(x1, x2, y1, y2)
    # b1 = b1.crop(x1, x2, y1, y2)

    # dbiasff1 = ff1-b1  # debiased FF1
    # dbiasff2 = ff2-b1  # debiased FF2
    signal = (dbiasff1+dbiasff2)/2.0  # mean signal
    shotnoise = dbiasff1-dbiasff2

    orig_signal = signal.mean(RETURN=True, AXIS=1)
    orig_var = shotnoise.var(RETURN=True, AXIS=1)/2.0

    # Shot Noise data
    shotnoisedata = shotnoise.get_data()
    # Compute Z score
    zsco = stats.zscore(shotnoisedata, axis=1, ddof=1)
    # Create mask for values outside +3/-3 std
    snarray_mask = np.ma.masked_outside(zsco, -1.0*nstd, nstd)
    sn_masked = np.ma.masked_where(np.ma.getmask(snarray_mask), shotnoise.get_data())
    shotnoise.data = sn_masked

    signalmean = signal.mean(RETURN=True, AXIS=axis)
    variance = shotnoise.var(RETURN=True, AXIS=axis)/2.0
    cf = signalmean/variance

    # fit line using all data (later we eliminate wrong values by masking )
    coefts = np.polyfit(signalmean, variance, order)
    polycoef = np.poly1d(coefts)
    var_fitted = np.polyval(polycoef, signalmean[:])

    # if RETURN equal True, return signal_masked, variance masked, fitted variance and CF
    if kargs.get('RETURN', False):
        if order == 1:
            return signalmean, variance, var_fitted, (1/coefts[0])
        else:
            return signalmean, variance, var_fitted, (1/coefts[1])

    else:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10, 5))
        # plot variance vs signal without masking yet
        # ax1.plot(orig_signal, orig_var, '.b')
        ax1.plot(signalmean, cf, '.b')
        ax1.grid()
        ax1.set_title('Photon Transfer Curve')
        ax1.set_ylabel('Variance [ADU]**2')
        ax1.set_xlabel('Signal [ADU]')

        # plot variance vs signal WITH masking
        ax2.plot(signalmean, variance, '.b', signalmean, var_fitted, 'r')
        ax2.grid()
        title = 'PTC  CF=%f'
        if order == 2:
            title = title % (1/coefts[1])
        else:
            title = title % (1/coefts[0])
        ax2.set_title(title)
        ax2.set_ylabel('Variance [ADU]**2')
        ax2.set_xlabel('Signal [ADU]')


def ptc_2ff(bias, ff1, ff2, *coor, **kargs):
    """
    Perform ptc plot with one bias and 2 tdi images.
    ex:
    ptc_2tdi(b1,tdi1,tdi2,50, 2000, 100, 170)
    compute CF using 2 bias and 2 FF in an area defined by [50:2000,100:170] plot the ptc curve and compute the CF using
    a first order polynomia

    ptc_2tdi(b1,tdi1,tdi2,50, 2000, 100, 170, RETURN=True)
    compute CF using 2 bias and 2 FF in an area defined by [50:2000,100:170] return the vectors and the CF, using
    a first order polynomia

    ptc_2tdi(b1,tdi1,tdi2,50, 2000, 100, 170, ORDER=2)
    compute CF using 2 bias and 2 FF in an area defined by [50:2000,100:170] plot the ptc curve, and use a
    polynomia of order 2


    The 2 tdi images should have a slope in flux to compute the ptc.
    To eliminate the FPN, the 'shotnoise' image is computed as the subtraction
    of two debiased flat field images
    optional kargs arguments:
    NSTD (default = 3) Default number of std deviation to elliminate outlayers
    VERBOSE (default=False)

    This method can be used on the TestBench
    """

    nstd = kargs.get('NSTD', 3)  # factor to elliminate outlayers
    order = kargs.get('ORDER', 2)
    axis = kargs.get('AXIS', 1)  # columns

    #dbiasff1, dbiasff2 = correctTDIShift(bias, tdi1, tdi2)
    dbiasff1 = ff1 - bias
    dbiasff2 = ff2 - bias

    x1, x2, y1, y2 = dbiasff1.get_windowcoor(*coor)

    dbiasff1 = dbiasff1.crop(x1, x2, y1, y2)
    dbiasff2 = dbiasff2.crop(x1, x2, y1, y2)
    # b1 = b1.crop(x1, x2, y1, y2)

    # dbiasff1 = ff1-b1  # debiased FF1
    # dbiasff2 = ff2-b1  # debiased FF2
    signal = (dbiasff1+dbiasff2)/2.0  # mean signal
    shotnoise = dbiasff1-dbiasff2

    orig_signal = signal.mean(RETURN=True, AXIS=1)
    orig_var = shotnoise.var(RETURN=True, AXIS=1)/2.0

    # Shot Noise data
    shotnoisedata = shotnoise.get_data()
    # Compute Z score
    zsco = stats.zscore(shotnoisedata, axis=1, ddof=1)
    # Create mask for values outside +3/-3 std
    snarray_mask = np.ma.masked_outside(zsco, -1.0*nstd, nstd)
    sn_masked = np.ma.masked_where(np.ma.getmask(snarray_mask), shotnoise.get_data())
    shotnoise.data = sn_masked

    signalmean = signal.mean(RETURN=True, AXIS=axis)
    variance = shotnoise.var(RETURN=True, AXIS=axis)/2.0
    cf = signalmean/variance

    # fit line using all data (later we eliminate wrong values by masking )
    coefts = np.polyfit(signalmean, variance, order)
    polycoef = np.poly1d(coefts)
    var_fitted = np.polyval(polycoef, signalmean[:])

    # if RETURN equal True, return signal_masked, variance masked, fitted variance and CF
    if kargs.get('RETURN', False):
        if order == 1:
            return signalmean, variance, var_fitted, (1/coefts[0])
        else:
            return signalmean, variance, var_fitted, (1/coefts[1])

    else:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10, 5))
        # plot variance vs signal without masking yet
        # ax1.plot(orig_signal, orig_var, '.b')
        ax1.plot(signalmean, cf, '.b')
        ax1.grid()
        ax1.set_title('Photon Transfer Curve')
        ax1.set_ylabel('Variance [ADU]**2')
        ax1.set_xlabel('Signal [ADU]')

        # plot variance vs signal WITH masking
        ax2.plot(signalmean, variance, '.b', signalmean, var_fitted, 'r')
        ax2.grid()
        title = 'PTC  CF=%f'
        if order == 2:
            title = title % (1/coefts[1])
        else:
            title = title % (1/coefts[0])
        ax2.set_title(title)
        ax2.set_ylabel('Variance [ADU]**2')
        ax2.set_xlabel('Signal [ADU]')


def ptc_pixels(biaslist, fflist, ext=0, *coor, **kargs):
    """
    Perform ptc computation to get gain and RON from a list of bias names and a
    list of ff images names.
    The ff images should be the same scene with all possible light levels.
    An example would be a grism ff on FORS or MUSE
    To eliminate the FPN, the analysis is done pixel by pixel.

    optional kargs:
    LOW: low level in ADUs to compute ptc
    HIGH: high level in ADU to compute ptc
    STEP: step in ADUs to compute variance (small steps will slow down the computation)
    ORDER (default = 1) polynomial order to fit the ptc
    RETURN (default=FALSE) returns CF and RON

    Ex:
    ptc_pixels(biaslist, fflst, 0,2000,300,600, LOW=100, HIGH=50000, STEP=10)

    signal, var, var_fitted, cf = ptc_pixels(
        b1, ff1, ff2, 100,200,10, 2000, LOW=100, HIGH=50000, STEP=100, OLAYERS=0.4, RETURN=True)
    Compute the PTC in the window [100:200,10:2000] from 100ADUs up to 50000 ADUs each 100 ADUs,

    """

    low = kargs.get('LOW', 0)  # minimum signal level to explore
    high = kargs.get('HIGH', 60000)  # maximum signal level to explore
    step = kargs.get('STEP', 100)  # step size, minimum is 1
    nwx = kargs.get('NWX', 10)  # size of windows in X to compute RON
    nwy = kargs.get('NWY', 10)  # size of windows in Y to compute RON

    # print("Low = {}".format(low))
    # print("High = {}".format(high))
    # print("Step = {}".format(step))

    order = kargs.get('ORDER', 1)  # order of polynomial regression
    if order > 2:
        order = 2

    # print("Order = {}".format(order))

    # read biaslist
    biasimages = [Image(i, ext) for i in biaslist]

    # read fflist
    ffimages = [Image(i, ext) for i in fflist]

    x1, x2, y1, y2 = biasimages[0].get_windowcoor(*coor)

    # print("{},{},{},{}".format(x1,x2,y1,y2))

    # crop bias images
    biascroped = [i.crop(*coor) for i in biasimages]

    # compute bias mean
    biasmean = meanstack(biascroped)
    # print(biasmean.shape)

    # compute
    stdsig = np.zeros((nwx, nwy))
    windows = subwindowcoor((x2-x1)//2-5*nwx, (x2-x1)//2+5*nwx,
                            (y2-y1)//2-5*nwy, (y2-y1)//2+5*nwy, **kargs)
    for i, j, xi, xf, yi, yf in windows:
        stdsig[i, j] = biasmean[xi:xf, yi:yf].std() * np.sqrt(len(biascroped))

    # crop ff images
    ffcroped = [i.crop(*coor) for i in ffimages]
    # print(ffcroped[0].shape)

    # debiased ff
    ffcropdb = [(i-biasmean) for i in ffcroped]

    ffcropdb_data = [i.get_data() for i in ffcropdb]

    # compute signal
    ffsignal = meanstack(ffcropdb)

    # use only data from images
    ffsignal_data = ffsignal.get_data()

    # compute variance of all ffcropdb images along axis 0
    ffcropdb_stacked = np.stack(ffcropdb_data)
    ffvar = ffcropdb_stacked.var(ddof=1, axis=0)

    # flatten resulting arrays
    ffsignal_flatten = ffsignal_data.flatten()
    ffvar_flatten = ffvar.flatten()
    # print("ffsignal_flatten, ffvar_flatten : {}, {}".format(len(ffsignal_flatten), len(ffvar_flatten)))

    # convert ffsignal_flatten in integer
    ffsignal_flatten = ffsignal_flatten.astype(int)  # [int(i) for i in ffsignal_flatten]
    ffvar_flatten = ffvar_flatten.astype(int)

    # sort ffsignal_flatten and  ffvar_flatten
    # indx = np.argsort(ffsignal_flatten)
    # ffsignal_flatten = ffsignal_flatten[indx]
    # ffvar_flatten =ffvar_flatten[indx]

    # get unique values in ff
    ffsignal_unique = np.unique(ffsignal_flatten)
    # print("ffsignal_unique : {}".format(len(ffsignal_unique)))

    # filter out  values lower than LOW and higher than HIGH
    ffsig_unique = [i for i in ffsignal_unique if i >= low and i <= high]
    # print("ffsig_unique: {}".format(len(ffsig_unique)))

    # generate sampling values
    sampling = list(range(low, len(ffsig_unique), step))
    # print("sampling ={}".format(len(sampling)))

    # create subset of unique values using sampling
    ffsampled = [ffsig_unique[i] for i in sampling]
    # print("ffsampled : {}".format(len(ffsampled)))

    # by default use mean computation for variance
    if kargs.get('MEDIAN', False):
        # Compute variance mean for values in ffsignalflatten that are in ffsampled
        variance = [np.median(ffvar_flatten[ffsignal_flatten == i]) for i in ffsampled]
    else:
        variance = [np.mean(ffvar_flatten[ffsignal_flatten == i]) for i in ffsampled]


# filter out the pixels with variance==0 or greater than 200000
    variance = np.array(variance)
    ffsampled = np.array(ffsampled)
    vfiltered_index = np.where((variance > 0) & (variance < 200000))
    ffsampled = ffsampled[vfiltered_index]
    variance = variance[vfiltered_index]

    # print(len(ffsignal_unique), len(variance))

    plt.scatter(ffsampled[::], variance[::])
    plt.grid()
    plt.show()

    # compute polynomial without filtering outlayers
    coefts_nf = np.polyfit(ffsampled, variance, order)
    polyts_nf = np.poly1d(coefts_nf)
    # var_fitted = polyts_nf(ffsampled)

    if order == 2:
        gain = 1/coefts_nf[1]
        print("GAIN = {} -e/ADU  RON = {} -e".format(1/coefts_nf[1], gain*np.median(stdsig)))
    else:
        gain = 1/coefts_nf[0]
        print("GAIN = {} -e/ADU  RON = {} -e".format(1/coefts_nf[0], gain*np.median(stdsig)))

    # ron = gain*np.median(stdsig)
    # print("RON = {} e".format(ron))


def ptc_2pixels(b1, ff1, ff2, *coor, **kargs):
    """
    Perform ptc computation using 2ff and one bias but using pixel analysis
    The 2 ff images should be the same scene with all possible light levels,
    for example a grism FF in FORS or FF in MUSE.
    To eliminate the FPN, the analysis is done pixel by pixel

    optional kargs arguments:
    LOW: low level in ADUs to compute ptc
    HIGH: high level in ADU to compute ptc
    STEP: step in ADUs to compute variance (small steps will slow down the computation)
    ORDER (default = 1) polynomial order to fit the ptc
    OLAYERS (default = 0.5)


    Ex:
    ptc_pixels(b1, ff1, ff2, 100,200,10, 2000, LOW=100, HIGH=50000, STEP=100, OLAYERS=0.4)
    if RETURN=True  the function will return the signal and variance and fitted variance array plus the CF
    signal, var, var_fitted, cf = ptc_pixels(
        b1, ff1, ff2, 100,200,10, 2000, LOW=100, HIGH=50000, STEP=100, OLAYERS=0.4, RETURN=True)
    Compute the PTC in the window [100:200,10:2000] from 100ADUs up to 50000 ADUs each 100 ADUs,

    """

    low = kargs.get('LOW', 0)  # minimum signal level to explore
    high = kargs.get('HIGH', 60000)  # maximum signal level to explore
    step = kargs.get('STEP', 100)  # step size, minimum is 1
    outlayers = kargs.get('OLAYERS', 0.5)  # factor to elliminate outlayers

    print("Low = {}".format(low))
    print("High = {}".format(high))
    print("Step = {}".format(step))
    print("outlayers = {}".format(outlayers))

    order = kargs.get('ORDER', 1)  # order of polynomial regression
    if order > 2:
        order = 2

    print("Order = {}".format(order))

    x1, x2, y1, y2 = b1.get_windowcoor(*coor)

    print("{},{},{},{}".format(x1, x2, y1, y2))

    # crop images
    ff1c = ff1.crop(x1, x2, y1, y2)
    ff2c = ff2.crop(x1, x2, y1, y2)
    bias1c = b1.crop(x1, x2, y1, y2)

    # remove FPN diferentiating 2 FF
    noise = ff1c-ff2c
    signal = ff1c-bias1c

    # noised contains the difference between both ff, from here we will calculate the variance
    noised = noise.get_data()  # .astype(int16)

    signald = signal.get_data()  # .astype(int16)

    # make a list with all the discrete values in the window
    s_unique = np.unique(signald)
    # now use only those between 'low' and 'high' level
    s_unique = s_unique[(s_unique <= high) & (s_unique >= low)]
    # to diminish computing time, select only some of the values for computing the ptc
    # first generate a list with the indexes to use
    sampling = list(range(0, len(s_unique), step))
    # Now extract from s_unique only the levels asociated with the indexes
    s = s_unique[sampling]

    # v =[np.ma.masked_where(signald!=i,noised).var()/2.0 for i in s ]    #This is quite sloooow

    # compute the variance for all the values in the noised array where the signal is equal to i (for all i values in the array s),
    # devide it by 2 as noised variance is larger by that factor due to subtraction
    # this way is much faster than masking
    v = [np.var(noised[signald == i], ddof=1)/2.0 for i in s]
    # v =[np.var(noised[signald==i])/2.0 for i in s]    #this way is much faster than masking

    s = np.array(s)
    v = np.array(v)

    # filter out the pixels with variance==0
    vnonzero_index = np.where(v != 0)
    snz = s[vnonzero_index]
    vnz = v[vnonzero_index]

    # compute polynomial without filtering outlayers
    coefts_nf = np.polyfit(snz, vnz, order)
    polyts_nf = np.poly1d(coefts_nf)
    var_fitted = np.polyval(polyts_nf, snz)

    # line or curve defining max variance to use
    v_lim_sup = var_fitted + outlayers * var_fitted
    # line or curve defining min variance to use
    v_lim_inf = var_fitted - outlayers * var_fitted

    # mask out all pixels variance wich is above or below limits
    var_masked = np.ma.masked_where((vnz > v_lim_sup) | (vnz < v_lim_inf), vnz)
    # mask out all pixels wich correspond to variance above or below limits
    sig_masked = np.ma.masked_where((vnz > v_lim_sup) | (vnz < v_lim_inf), snz)

    # reduce size of arrays to good pixels
    var_compressed = np.ma.compressed(var_masked)
    sig_compressed = np.ma.compressed(sig_masked)

    # do curve fitting (of order 1 or 2)
    coefts = np.polyfit(sig_compressed, var_compressed, order)
    polyts = np.poly1d(coefts)
    var_compressed_fitted = np.polyval(polyts, sig_compressed)

    if kargs.get('RETURN', False):
        if order == 2:
            return sig_compressed, var_compressed, var_compressed_fitted, (1/coefts[1])
        else:
            return sig_compressed, var_compressed, var_compressed_fitted, (1/coefts[0])

    else:
        # Use subplots to plot before and after filtering
        # plot with outlayers and finally filtering the outlayers
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10, 5))
        ax1.plot(snz, vnz, '.b', snz, v_lim_sup, 'r', snz, v_lim_inf, 'r')
        ax1.set_xlabel('Signal [ADU]')
        ax1.set_ylabel('Variance [ADU]**2')

        if order == 2:
            ax1.set_title('CF = %f' % (1/coefts_nf[1]))
        else:
            ax1.set_title('CF = %f' % (1/coefts_nf[0]))

        ax1.grid()
        ax2.plot(sig_compressed, var_compressed, '.b', sig_compressed, var_compressed_fitted, 'r',)
        ax2.grid()

        if order == 2:
            ax2.set_title('CF = %f' % (1/coefts[1]))
        else:
            ax2.set_title('CF = %f' % (1/coefts[0]))
        ax2.set_xlabel('Signal [ADU]')
        ax2.set_ylabel('Variance [ADU]**2')


# TODO: Implement ptc curve for IR sensors using many ff at different signal levels
def ptc_ir(imagelist, variancelist,  NDIT=10, *coor, **kargs):
    """
    ptc computation for IR sensors
    This ptc is computed from many groups of files, with same DIT until reaching saturation.
    For the infrared detector using IRACE and NGC, it's possible to get NDIT images for each DIT
    and also store the variance for each NDIT images.
    For example 30 images at DIT=1, then 30 at DIT=2, etc...


    """
    # x1,x2,y1,y2=bias1.get_windowcoor(*coor)
    # TODO continue with the program for this routine


def ptc_irffpairs(imagelist, *coor, **kargs):
    """
    TODO: Need to be finished !!
    NHA

    Perform ptc plot for pairs of ff at same level.
    The pairs of ff should have the same light level.
    The first 2 images must be bias
    To eliminate the FPN, the 'shotnoise' image is computed as the subtraction
    of two debiased flat field images
    optional kargs arguments:
    FACTOR (default = 2.0)
    FIRST_FITTING (default = False)
    LIMIT (default = False)
    VERBOSE (default=False)
    TODO: add Rotation by 90deg as an option
    """
    x1, x2, y1, y2 = imagelist[0].get_windowcoor(*coor)

    # Define empty list to store values
    signal = []
    stddev = []
    variance = []

    oddimageindex = list(range(1, len(imagelist), 2))
    evenimageindex = list(range(0, len(imagelist), 2))

    # For all pairs, compute signal, std and variance
    for odd, even in zip(oddimageindex, evenimageindex):
        ff1 = imagelist[odd]
        ff2 = imagelist[even]
        ffmean = (ff1 + ff2)/2.0
        shotnoise = ff1 - ff2
        signal.append(np.mean(ffmean[x1:x2, y1:y2]))
        variance.append(np.var(shotnoise[x1:x2, y1:y2])/2.0)
        stddev.append(np.std(shotnoise[x1:x2, y1:y2])/np.sqrt(2.0))
        print("Signal: %f   Variance: %f" % (signal[-1], variance[-1]))

    coefts = ma.polyfit(signal, variance, 1)
    # coefts=ma.polyfit(signal[:-3],variance[:-3],1)
    polyts = coefts[0]*np.array(signal)+coefts[1]
    print(1/coefts[0], coefts[1])

    # plot variance v/s signal
    plt.figure()
    # plot(meansig,masked_variance,'b.')
    plt.plot(signal, variance, 'b.')
    plt.plot(signal, polyts, 'r-')

    # Plot the curves

    # compute the CF


# TODO: Implement the ptc curve using ff images at different levels
#      in loglog scale to show the different ptc zones
def ptcloglog(imagelist, *coor, **kargs):
    """
    TODO: Need to be finished!!

    Perform ptc plot for ff at different light levels

    The first image must be a bias

    The FPN is not elliminated. This curve should illustrate the diferent regions of a
    tipical detector

    """
    signal = []
    variance = []
    stddev = []

    x1, x2, y1, y2 = imagelist[0].get_windowcoor(*coor)
    print("No error")

    # Read in bias
    bias = imagelist[0]

    # For all images, compute signal, std and variance
    for image in imagelist[1:]:
        ff = image
        ff = ff - bias

        signal.append(np.mean(ff[x1:x2, y1:y2]))
        variance.append(np.var(ff[x1:x2, y1:y2]))
        stddev.append(np.std(ff[x1:x2, y1:y2]))
        print("Signal: %f   Variance: %f" % (signal[-1], variance[-1]))

    coefts = ma.polyfit(signal[:-3], variance[:-3], 1)
    polyts = coefts[0]*signal+coefts[1]
    print(1/coefts[0], coefts[1])

    # plot variance v/s signal
    plt.figure()
    # plot(meansig,masked_variance,'b.')
    # plot(signal, variance, 'b.')
    # plot(signal[:-3],polyts[:-3],'r-')

    # Plot the curves
    plt.loglog(signal, stddev, 'r')

    # compute the CF
