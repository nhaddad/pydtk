#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:22:21 2018

@author: nhaddad
"""
from astropy.io import fits as pyfits
import os
import re
import glob
from pathlib import Path
import numpy as np
from pydtk import Image
import matplotlib.pyplot as plt
import pandas as pd





def generate_fpn_pixel(**kargs):
    """
    Generates a matrix with random coeficients that introduces a fix pattern noise
    to the detector pixel to pixel response
    Ex:
    fpnp = generate_fpn_pixel(NROWS=300, NCOLS=400, VALUE=0.01)
    The above command will generate a random matrix of 300x400 pixels and
    with values ranging from 1.0 +/- VALUE  with a normal (gaussian distribution)
    Note: maximum VALUE=0.15



    """
    value = kargs.get('VALUE', 0.01)
    if value>0.15:
        value = 0.15
    nrows = kargs.get('NROWS', 512)
    ncols = kargs.get('NCOLS', 512)
    fpnp = np.random.normal( 1, value, nrows * ncols)
    return fpnp.reshape(nrows, ncols)

def generate_fpn_row(**kargs):
    """
    Generates a vector with random coeficients that introduces a fix pattern noise
    to the detector pixel to pixel response
    Ex:
    fpnr = generate_fpn_row(NROWS=300, VALUE=0.01)
    The above command will generate a random vector of 300x1 pixels and
    with values ranging from 1.0 +/- VALUE  with a normal (gaussian distribution)
    Note: maximum VALUE=0.15


    """
    value = kargs.get('VALUE', 0.01)
    if value>0.15:
        value = 0.15

    nrows = kargs.get('NROWS', 512)
    fpnr = np.random.normal( 1, value, nrows )
    return fpnr.reshape(nrows,1)



def generate_fpn_col(**kargs):
    """
    Generates a vector with random coeficients that introduces a fix pattern noise
    to the detector pixel to pixel response
    Ex:
    fpnc = generate_fpn_col(NCOLS=400, VALUE=0.01)
    The above command will generate a random vector of 1x300 pixels and
    with values ranging from 1.0 +/- VALUE  with a normal (gaussian distribution)
    Note: maximum VALUE=0.15


    """
    value = kargs.get('VALUE', 0.01)
    if value>0.15:
        value = 0.15

    ncols = kargs.get('NCOLS', 512)
    fpnc = np.random.normal( 1, value, ncols )
    return fpnc #.reshape(ncols,1)




def infofits(filepath, **kargs):
    '''
    Syntax:
    infofits('filename.fits')
        infofits('filename.fits', RETURN=True)

    This function returns information of extensions in a FITS file
    fits.info


    Example:
    infofits('FORS2_IMG_CAL099.80.CHIP1.fits') which gives, for this file

    Filename: FORS2_IMG_CAL099.80.CHIP1.fits
    No.    Name         Type      Cards   Dimensions   Format
    0    CHIP1       PrimaryHDU     242   (2048, 1034)   int16

    For an OmegaCAM image we get

    infofits('OMEGACAM_IMG_FLAT120_0020.fits')

    Filename: OMEGACAM_IMG_FLAT120_0020.fits
    No.    Name         Type      Cards   Dimensions   Format
    0    PRIMARY     PrimaryHDU     612   ()
    1    ESO_CCD_#65  ImageHDU        52   (2144, 4200)   int16 (rescales to uint16)
    2    ESO_CCD_#66  ImageHDU        52   (2144, 4200)   int16 (rescales to uint16)
    3    ESO_CCD_#67  ImageHDU        52   (2144, 4200)   int16 (rescales to uint16)
    4    ESO_CCD_#68  ImageHDU        52   (2144, 4200)   int16 (rescales to uint16)
    5    ESO_CCD_#73  ImageHDU        52   (2144, 4200)   int16 (rescales to uint16)
    6    ESO_CCD_#74  ImageHDU        52   (2144, 4200)   int16 (rescales to uint16)
    ....etc (32 CCDs)


    '''
    if Path(filepath).is_file():
        if kargs.get('RETURN',False):
            return pyfits.info(filepath, output=False)
        else:
            pyfits.info(filepath)

    else:
        print(f'File "{filepath}" not exist')

def get_ext_list(filename, substr):
    '''
    Use fits.info to get a list of all the extensions in the image file
    ext_list = get_ext_list(filename, substr)
    ext_list = get_ext_list('MUSE_001.fits', 'CHAN')

    '''
    if Path(filename).is_file():
        info = pyfits.info(filename, output=False)
        ext_list = [i[1] for i in info if (substr in i[1])]
        return ext_list
    else:
        print(f'File "{filename}" not exist')




def get_datacube(filename, ext_list):
    '''
    For images containing multiple extension (VIRCAM, OMEGACAM, etc), this utility function
    generate a data cube of numpy arrays
    ext_list is a list of valid extensions to read like:
        [1,2,3]
        ['CHAN01', 'CHAN02', 'CHAN24']

    '''
    if Path(filename).is_file():
        data =[]
        hdu = pyfits.open(filename)
        for ext in ext_list:
            data.append(hdu[ext].data.astype(float))
        return np.array(data)
    else:
        print(f'File "{filename}" not exis')


def getheader(filename, ext=0, FILTER=None, **kargs):
    """
    Read the header for a FITS file, if a FILTER string is specified,
    the output will contain only lines that have FILTER in the keyword
    if ONCOMMENTS=True, then also look into comments


    Example:
    getheader('FORS2_IMG_CAL099.80.CHIP1.fits','CHIP1') or
    getheader('FORS2_IMG_CAL099.80.CHIP1.fits',0)
    getheader('FORS2_IMG_CAL099.80.CHIP1.fits',0, FILTER=' DET ')   #this will print only
    header lines containing detector keywords
    getheader('FORS2_IMG_CAL099.80.CHIP1.fits',0, FILTER='gain', ONCOMMENTS=True)   #this will print only
    header and comment lines containing the word 'gain'

    To get the available extensions use:
    infofits('FORS2_IMG_CAL099.80.CHIP1.fits') which gives, for this file

    Filename: FORS2_IMG_CAL099.80.CHIP1.fits
    No.    Name         Type      Cards   Dimensions   Format
    0    CHIP1       PrimaryHDU     242   (2048, 1034)   int16


    """

    try:
        header = pyfits.getheader(filename,ext)
        keys = [i for i in list(header.keys()) if i!='']
        dict_out = { i:[header.get(i), header.comments[i]] for i in keys}


    except:
        print("Error opening the filename or wrog extension")
        return None

    #if FILTER is None, print all the non empty keywords
    if FILTER is None:
        if kargs.get('RETURN',False):
            print(pd.DataFrame.from_dict(dict_out, orient='index', columns=['value', 'comments']))

        else:
            return pd.DataFrame.from_dict(dict_out, orient='index', columns=['value', 'comments'])

    #TODO: check for FILTER also in values and comments
    elif isinstance(FILTER,str):
        FILTER = FILTER.upper()     #set FILTER to uppercase
        #get all the keys not empty
        keys = [i for i in list(header.keys()) if i!='']

        #from keys, generate a subset of keys called 'subkeys' which contains the keys that have FILTER or the comments that have FILTER
        if kargs.get('ONCOMMENTS',True) :
            subkeys = [k for k in keys if (FILTER in k) or (FILTER in header.comments[k].upper())]
        else:
            subkeys = [k for k in keys if (FILTER in k)]

        #values = [header.get(key) for key in subkeys]
        #comments = [header.comments[key] for key in subkeys]
        #headerlist = list(zip(subkeys, values, comments))
        #headerlist = [line for line in headerlist if line.cards!=('','','')]
        dict_out = { i:(header.get(i), header.comments[i]) for i in subkeys}

        if kargs.get('RETURN',False): #if RETURN==False print to console
            print(pd.DataFrame.from_dict(dict_out, orient='index', columns=['value', 'comments']))#.transpose())

        else:                         #if RETURN==True
            return pd.DataFrame.from_dict(dict_out, orient='index', columns=['value', 'comments'])#.transpose()

    #else:
        #return None
    return None



def dfitsort(filefilter='*fits', filepath='.', FILTERLIST=None,  **kargs ):
    """
    Print a list with filename, Filters and values with the same format
    If you need to store the list, use RETURN=True
    as dfits *fits | fitsort Filter1 Filter2 ... etc
    Filter_i are regular expresions
    Note:
        '.*' means any character
        '^xxx' means begin with xxx
        'xxx$' means finish with xxx

    kargs:
    RETURN=[False]
    DF=[FALSE]  return a Pandas dataframe if true

    Examples:
    dfitsort(FILTERLIST=['^EXPTIME$'])
    List the keyword that start with EXPTIME and finish with EXPTIME
    this wil match 'EXPTIME'

    mylist = dfitsort(FILTERLIST=['^EXPTIME$'], RETURN=True)
    save list in mylist instead of printing out

    dfitsort(FILTERLIST=['.*EXPTIME$'])
    List keywords that start with any character but ends with EXPTIME
    this wil match 'EXPTIME' and 'ESO INS LAMP1 EXPTIME'

    dfitsort(FILTERLIST=['.*EXPTIME$','.*READ.*'])
    List keywords that start with any character and ends with EXPTIME and also
    all keywords that have READ in the middle
    this will match
    'ESO DET READ MODE',
    'ESO DET READ SPEED',
    'ESO DET READ CLOCK',
    'ESO DET READ NFRAM',


    """

    #TODO make a list of all filenames that fits 'filefilter'

    path = Path(filepath)
    filelist = list(path.glob(filefilter))
    #filelist = glob.glob(filefilter)
    longestname = len(max([i.name for i in filelist], key=len))


    ext = kargs.get('ext',0)

    output_dic = {}
    for filename in filelist:
        #Read header of file
        try:
            #print filename
            header = pyfits.getheader(filename,ext)
            keys = list(header.keys())
            keys = [k for k in keys if k!=''] #elliminate empty keys
        except:
            print("Error opening the filename or wrog extension")
            return None

        #Check each filter against header
        #values = []

        # Check if FILTERLIST is not empty
        if isinstance(FILTERLIST, list):
            subkeys = []
            for filt in FILTERLIST:
                rexp = re.compile(filt, re.IGNORECASE)
                    #subkeys = [k for k in keys if rexp.match(k)!=None]
                for k in keys:
                    if rexp.match(k)!=None:
                        subkeys.append(k)


        else:
            print('No FILTERLIST provided!')
            return None

        # check if subkey is not empty len(subkeys)>0:
        if subkeys:
            # Get all the keyword values as a dict
            values = {i:str(header.get(i)) for i in subkeys}
            # create an output dictionary with filename as key and all values as another dict
            output_dic.update({filename:values})

    # if RETURN=True, return a Pandas dataframe or a dictionary if
    # PANDAS=False
        pass
    if kargs.get('RETURN',True):
        if kargs.get('PANDAS',True):
            return pd.DataFrame.from_dict(output_dic, orient='index')#.transpose()
        else:
            return output_dic

    else:
        #TODO improve output format
        for key,value in output_dic.items():
            listval = [val for i,val in value.items()]
            listval.insert(0,key)
            listval_str = ' '.join(listval)
            print(listval_str)



    return None



#TODO convert masklessthan and maskgreaterthan in methods of class Image
def masklessthan(image, pixval):
    """
    Generate a masked array with all pixels with value less than pixval

    Syntax: m=masklessthan(image,pixval)


    example: m=masklessthan(image,1200.0)
    mask out all values in image less than 1200.0 ADUs


    """
    return np.ma.masked_where(image.data<pixval, image.data)

def maskgreaterthan(image, pixval):
    """
    Generate a masked array with all pixels with value less than pixval

    Syntax: m=maskgreaterthan(image,pixval)


    example: m=maskgreaterthan(image,1200.0)
    mask out all values in image greater than 1200.0 ADUs

    """
    return np.ma.masked_where(image.data>pixval, image.data)

def maskoutside(image,pixval_low,pixval_high):
    """
    Generate a masked array with all pixels with value less than pixval_low
    or greater than pixvalhigh

    Syntax: m=maskoutside(image,pixval_low,pixelval_high)


    example: m=maskoutside(image,1200.0,1600.0)
    mask out all values in image less than 1200.0 ADUs or greater than 1600.0

    """
    return np.ma.masked_outside(image.data, pixval_low, pixval_high)

def maskinside(image,pixval1,pixval2):
    """
    Generate a masked array with all pixels with value between pixval1 and pixval2

    Syntax: m=maskinside(image,pixval1,pixelval2)


    example: m=maskoutside(image,1200.0,1600.0)
    mask out all values in image but not including 1200 and 1600

    """
    return np.ma.masked_inside(image.data, pixval1, pixval2)

def maskoutside_std(image, FACTOR=3):
    """
    Generate a masked array with all pixels with value less or
    greater than STD*image(std)

    Syntax: m=maskoutside(image)


    example: m=maskoutside(image)

    mask out all values in image less or greater than mean+/- 3*std

    """
    mean = image.mean()
    std = image.std()
    return np.ma.masked_outside(image.data, mean-FACTOR*std, mean+FACTOR*std)




def windcoorinmesh(*coor, **kargs):
    """
    generate coordinates for windows inside n area
    Usage:
    win = windcoormesh(0,4000,0,1000, NWX=3, NWY=3)
    generates 9 windows

    TODO: Improve documentation on this

    """
    nwx = kargs.get('NWX', 3)    #set number of windows in x direction
    nwy = kargs.get('NWY', 3)    #set number of windows in y direction
    dwx = kargs.get('WIDTHX',100) #set half of window width in x direction
    dwy = kargs.get('WIDTHY',100) #set half of window width in y direction


    x1, x2, y1, y2 = coor

    wx = (x2-x1)//(nwx+1)    #size of every subwindow in x
    wy = (y2-y1)//(nwy+1)    #size of every subwindow in y

    for i in range(1,nwx+1):
        for j in range(1,nwy+1):
            xi= x1 + i*wx - dwx
            xf= x1 + i*wx + dwx
            yi= y1 + j*wy - dwy
            yf= y1 + j*wy + dwy
            yield(xi,xf,yi,yf)


def stackslices(ImageName, firstslice=0, lastslice=2):
    """
    Read the  image slices from firstslice to lastslice and return the numpy array

    """

def plotfft(imagelist,kps,vdelay, **option):
    """
    Plots the row and column fft for all images in imagelist

    All the images must have the same dimension and readout speed

    kps = read out speed in kilo pixels per sec
    vdelay = vertical delay between last row pixel and first next row pixel
    option parameters:
    BSTART (default = 3) is the bins to skip for finding the psd maxima

    example:
    plotfft([im1,im2,im3], 100, 5, FNAME='XSHOOTER 100kps')
    make a plot with fftrow and fftcol for each one of the images
    """

    #Check if imagelist is a list, if not convert it to one
    if not isinstance(imagelist,list):
        imagelist = [imagelist]


    subplotrow = len(imagelist)
    pdscol=[]   #create list to store computed column  power density spectra
    pdsrow=[]   #create list to store computed row power density spectra
    colmax=[]
    colmin=[]
    rowmax=[]
    rowmin=[]
    binstart = option.get('BSTART',0)
    filename = option.get('FNAME','')

    for im in imagelist:
        pdscol.append(im.fftcol(kps, vdelay, RETURN=True))
        colmax.append(pdscol[len(pdscol)-1][1][binstart:].max())  #look for pds maxima to set y limit
        colmin.append(pdscol[len(pdscol)-1][1][binstart:].min())  #look for pds minimum to set y limit
        pdsrow.append(im.fftrow(kps, PLOT=False))
        rowmax.append(pdsrow[len(pdsrow)-1][1][binstart:].max())  #look for pds maxima to set y limit
        rowmin.append(pdsrow[len(pdsrow)-1][1][binstart:].min())  #look for pds maxima to set y limit

    #look for Ymax and Ymin in fftcol
    ycolmax=colmax[colmax.index(max(colmax))]*1.05    #add 10% of headroom
    ycolmin=colmin[colmin.index(min(colmin))]*0.95    #add 10% of headroom

    #look for Ymax and Ymin in fftrow
    yrowmax=rowmax[rowmax.index(max(rowmax))]*1.05    #add 10% of headroom
    yrowmin=rowmin[rowmin.index(min(rowmin))]*0.95    #add 10% of headroom

    #print ycolmax,yrowmax

    fig = plt.figure(figsize=(12, subplotrow*3), dpi=100)
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(top=0.9)
    pltaxis=[]

    #TODO  Check if we need to use 'binstart' for plotting if we check the max from binstart
    #TODO  Add name of image on the left plot
    for i in range(subplotrow):
        #print firts fftcol on left column
        pltaxis.append(fig.add_subplot(subplotrow,2,i*2+1))
        pltaxis[len(pltaxis)-1].plot(pdscol[i][0][:],pdscol[i][1][:])
        pltaxis[len(pltaxis)-1].set_ylim(ycolmin,ycolmax)
        pltaxis[len(pltaxis)-1].grid()
        pltaxis[len(pltaxis)-1].set_title('Col FFT'+imagelist[i].filename, size=10)
        pltaxis[len(pltaxis)-1].tick_params(labelsize=8)

        #print now fftrow on right column
        pltaxis.append(fig.add_subplot(subplotrow,2,i*2+2))
        pltaxis[len(pltaxis)-1].plot(pdsrow[i][0][:],pdsrow[i][1][:])
        pltaxis[len(pltaxis)-1].set_ylim(yrowmin,yrowmax)
        pltaxis[len(pltaxis)-1].grid()
        pltaxis[len(pltaxis)-1].set_title('Row FFT'+imagelist[i].filename, size=10)
        pltaxis[len(pltaxis)-1].tick_params(labelsize=8)


    plt.suptitle('Column and Row FFT Analysis '+filename, size=16, y=1.05)
    plt.tight_layout()


def plotfftRatio(imagelist,refimage, kps,vdelay, **option):
    """
    Plots the row and column fft for all images in imagelist, devided by the refimage fft

    All the images must have the same dimension and readout speed

    kps = read out speed in kilo pixels per sec
    vdelay = vertical delay between last row pixel and first next row pixel
    option parameters:
    BSTART (default = 3) is the bins to skip for finding the psd maxima

    example:
    plotfft([im1,im2,im3], refim, 100, 5)
    make a plot with fftrow and fftcol for each one of the images
    """

    #Check if imagelist is a list, if not convert it to one
    if not isinstance(imagelist,list):
        imagelist = [imagelist]


    subplotrow = len(imagelist)   #compute how many rows will have the subplot
    refimagelist = [refimage]*len(imagelist) #generate a list repeating the refimage
    pdscol=[]   #create list to store computed column  power density spectra
    pdsrow=[]   #create list to store computed row power density spectra

    pdscolref = []
    pdsrowref = []

    ratiocol=[]  #empty list to store fftcol ration between images in imagelist and refimage
    ratiorow=[]  #empty list to store fftrow ration between images in imagelist and refimage

    colmax=[]
    colmin=[]
    rowmax=[]
    rowmin=[]
    ratiocolmax=[]
    ratiocolmin=[]
    ratiorowmax=[]
    ratiorowmin=[]

    binstart = option.get('BSTART',3)
    filename = option.get('FNAME','')


    for im,ref in zip(imagelist,refimagelist):

        pdscol.append(im.fftcol(kps, vdelay, RETURN=True))       #compute fft on column wise (low freq)
        colmax.append(pdscol[-1][1][binstart:].max())  #look for pds maxima to set y limit
        colmin.append(pdscol[-1][1][binstart:].min())  #look for pds minimum to set y limit
        pdsrow.append(im.fftrow(kps, PLOT=False))               #compute fft on row wise  (high freq)
        rowmax.append(pdsrow[-1][1][binstart:].max())  #look for pds maxima to set y limit
        rowmin.append(pdsrow[-1][1][binstart:].min())  #look for pds maxima to set y limit
        pdsrowref.append(ref.fftrow(kps, PLOT=False))
        ratiorow.append((pdsrowref[-1][0],np.asarray(pdsrow[-1][1])/np.asarray(pdsrowref[-1][1])))        #compute ration of row ffts  w/r refimage
        ratiorowmin.append(ratiorow[-1][1][binstart:].min())   #look for min in list to set y limit in plot
        ratiorowmax.append(ratiorow[-1][1][binstart:].max())   #look for max in list to set y limit in plot
        pdscolref.append(ref.fftcol(kps, vdelay, RETURN=True))
        ratiocol.append((pdscolref[-1][0],np.asarray(pdscol[-1][1])/np.asarray(pdscolref[-1][1])))        #compute ration of col ffts  w/r refimage
        ratiocolmin.append(ratiocol[-1][1][binstart:].min())   #look for min in list to set y limit in plot
        ratiocolmax.append(ratiocol[-1][1][binstart:].max())   #look for max in list to set y limit in plot

    #look for Ymax and Ymin in fftcol
    ycolmax=colmax[colmax.index(max(colmax))]*1.05    #add 10% of headroom
    ycolmin=colmin[colmin.index(min(colmin))]*0.95    #add 10% of headroom

    #look for Ymax and Ymin in fftrow
    yrowmax=rowmax[rowmax.index(max(rowmax))]*1.05    #add 10% of headroom
    yrowmin=rowmin[rowmin.index(min(rowmin))]*0.95    #add 10% of headroom

    #look for Ymax and Ymin in ratiocol
    ratiocolmax=ratiocolmax[ratiocolmax.index(max(ratiocolmax))]*1.05    #add 10% of headroom
    ratiocolmin=ratiocolmin[ratiocolmin.index(min(ratiocolmin))]*0.95    #add 10% of headroom

    #look for Ymax and Ymin in ratiorow
    ratiorowmax=ratiorowmax[ratiorowmax.index(max(ratiorowmax))]*1.05    #add 10% of headroom
    ratiorowmin=ratiorowmin[ratiorowmin.index(min(ratiorowmin))]*0.95    #add 10% of headroom



    #print ycolmax,yrowmax

    fig = plt.figure(figsize=(11, subplotrow*2.2), dpi=100)
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(top=0.9)
    pltaxis=[]

    #TODO  Check if we need to use 'binstart' for plotting if we check the max from binstart
    #TODO  Add name of image on the left plot
    for i in range(subplotrow):
        #print first fftcol on left column
        pltaxis.append(fig.add_subplot(subplotrow,2,i*2+1))
        pltaxis[len(pltaxis)-1].plot(ratiocol[i][0][:],ratiocol[i][1][:])
        pltaxis[len(pltaxis)-1].set_ylim(ratiocolmin,ratiocolmax)
        pltaxis[len(pltaxis)-1].grid()
        pltaxis[len(pltaxis)-1].set_title('ColFFT ratio '+imagelist[i].filename, size=10)
        pltaxis[len(pltaxis)-1].tick_params(labelsize=8)

        #print now fftrow on right column
        pltaxis.append(fig.add_subplot(subplotrow,2,i*2+2))
        pltaxis[len(pltaxis)-1].plot(ratiorow[i][0][:],ratiorow[i][1][:])
        pltaxis[len(pltaxis)-1].set_ylim(ratiorowmin,ratiorowmax)
        pltaxis[len(pltaxis)-1].grid()
        pltaxis[len(pltaxis)-1].set_title('RowFFT ratio '+imagelist[i].filename, size=10)
        pltaxis[len(pltaxis)-1].tick_params(labelsize=8)


    plt.suptitle('Col & Row FFT Ratio to '+refimage.filename, size=16, y=1.05)
    plt.tight_layout()


def medianstack(filelist, ext=0, **options):
    """
    Compute the median for a list of images.
    Usefull to elliminate cosmic rays from dark images
    Syntax:
        medianstack(filelist)
        return an image which is the median stack form the files in the current directory

    example:
    med = medianstack([dk1, dk2, dk3], 0)
    compute median for images dk1, dk2 and dk3 using extension 0

    TODO:
    Check if filelist is a list of images o list of strings and then perform
    the computation accordingly
    """


    #check if filelist is a list, if not raise error
    try:
        if (isinstance(filelist,list) and len(filelist)>1):
            pass
        else:
            raise Exception

    except:
        print("Not a list or len < 2")
        return None


    #check if elements in filelist is Image or string

    if all([isinstance(i,Image) for i in filelist]):
        imagesdata = [i.get_data() for i in filelist]
        im = filelist[0].copy()
        im.filename = 'medianstack'
        im.data=np.median(imagesdata, axis = 0)
        return im


    if all([isinstance(i,str) for i in filelist]):
        #copy first image on list to get same dim
        imagesdata = [Image(i,ext).get_data() for i in filelist]
        im = Image(filelist[0],ext)
        im.filename = 'medianstack'
        im.data=np.median(imagesdata, axis = 0)
        return im
    elif all([isinstance(i, Path) for i in filelist]):
        #copy first image on list to get same dim
        imagesdata = [Image(i,ext).get_data() for i in filelist]
        im = Image(filelist[0],ext)
        im.filename = 'medianstack'
        im.data=np.median(imagesdata, axis = 0)
        return im

    return None



def meanstack(filelist, ext=0, **options):
    """
    Compute the mean for a list of images.
    Syntax:
        medianstack(filelist)
        return an image which is the median stack form the files in the current directory

    example:
    med = meanstack([dk1, dk2, dk3], 0)
    compute median for images dk1, dk2 and dk3 using extension 0
    TODO:
    Check if filelist is a list of images o list of strings and then perform
    the computation accordingly

    """

    #check if filelist is a list, if not raise error
    try:
        if (isinstance(filelist,list) and len(filelist)>1):
            pass
        else:
            raise Exception

    except:
        print("Not a list or len < 2")
        return None


    #check if elements in filelist is Image or string
    if all([isinstance(i,Image) for i in filelist]):
        imagesdata = [i.get_data() for i in filelist]
        im = filelist[0].copy()
        im.filename = 'meanstack'
        im.data=np.mean(imagesdata, axis = 0)
        return im
    #check if filelist are files names

    if all([isinstance(i,str) for i in filelist]):
        #copy first image on list to get same dim
        imagesdata = [Image(i,ext).get_data() for i in filelist]
        im = Image(filelist[0],ext)
        im.filename = 'meanstack'
        im.data=np.mean(imagesdata, axis = 0)
        return im
    elif all([isinstance(i, Path) for i in filelist]):
        #copy first image on list to get same dim
        imagesdata = [Image(i,ext).get_data() for i in filelist]
        im = Image(filelist[0],ext)
        im.filename = 'meanstack'
        im.data=np.mean(imagesdata, axis = 0)
        return im


    return None


def stdstack(filelist, ext=0, **options):
    """
    Compute the standard deviation in z direction for a list of images.
    Syntax:
        stdstack(filelist)
        return an image which is the std pixel by pixel for the files in the list

    example:
    stdmap = stdstack([dk1, dk2, dk3,..,dkn], 0)
    compute median for images dk1, dk2, dk3 to dkn using extension 0


    """

    #check if filelist is a list, if not raise error
    try:
        if (isinstance(filelist,list) and len(filelist)>1):
            pass
        else:
            raise Exception

    except:
        print("Not a list or len < 2")
        return None


    #check if elements in filelist are Images or string

    if all([isinstance(i,Image) for i in filelist]):
        imagesdata = [i.get_data() for i in filelist]
        im = filelist[0].copy()
        im.filename = 'stdstack'
        im.data=np.std(imagesdata, axis = 0)
        return im
    #check if filelist are files names

    if all([isinstance(i,str) for i in filelist]):
        #copy first image on list to get same dim
        imagesdata = [Image(i,ext).get_data() for i in filelist]
        im = Image(filelist[0],ext)
        im.filename = 'stdstack'
        im.data=np.std(imagesdata, axis = 0)
        return im
    elif all([isinstance(i,Path) for i in filelist]):
        #copy first image on list to get same dim
        imagesdata = [Image(i,ext).get_data() for i in filelist]
        im = Image(filelist[0],ext)
        im.filename = 'stdstack'
        im.data=np.std(imagesdata, axis = 0)
        return im

    return None

def spectralangle(x0, y0, x1, y1):
    """
    Returns the angle in radian for a tilted spectra like UVES
    (x0,y0) => coordinates of lower point in a line that pass through
    the left side of the slit


    angle = spectraAngle(1000, 0, 1100, 2000)
    the result should be aprox 1.52

    """
    return np.arctan((y1-y0)/(x1-x0))


def getxy_tiltedspectra(x0, y0, y, angle=1.518):
    '''
    x0, y0: coordinates of lower left point of the line that pass through middle of slit
    y is the known line position
    x is the calculated position

    '''
    x = int((1/np.tan(angle))*(y - y0) + x0)

    return x,y

def extractchannel(filelist, channel):
    """
    extractchannel is a utility to read an extension from a multi
    extension fits file and save it as a fits file.

    It's usefull if we want to extract only one channel from,
    for example, a list of MUSE files.

    Usage:
    1)single case example
    extractchannel('MUSE_WFM_FLAT303_0031.fits','CHAN04')
    extract CHAN04 extension and create MUSE_WFM_FLAT303_0031_CHAN04.fits

    2)multi file example
    extractchannel('listing.txt','CHAN04')

    listing.txt is a text file with the listing of all files we want to
    extract the channel, it can be generated with the 'ls' command, like
    ls MUSE*WFM*fits > listing.txt

    3)filelist is python list with files names
    ex: flist=['OMEGACAM_100.fits', 'OMEGACAM_102.fits', 'OMEGACAM_103.fits']
    extractchannel(filelist, 'CCD_78')

    TODO: add OUTPATH which points to the directory where the extracted channels will be saved

    """

    #check if filelist is a fits file name
    if isinstance(filelist,str) and filelist.upper().endswith('.FITS'):
        try:
            im = Image(filelist, channel)
            im.save(filelist[:-5]+'_'+channel+'.fits')
        except:
            print('Bad channel designator')

    #check if filelist is a file
    elif os.path.isfile(filelist):
        files = open(filelist, 'r').readlines()
        for lines in files:
            try:
                #need to remove /n and split() generate a list, so
                #take first element
                im = Image(lines.split()[0], channel)
                im.save(lines[:-5]+'_'+channel+'.fits')
            except:
                print('Bad channel designator')

    #check if filelist is a python list of strings finished in fits
    elif isinstance(filelist,list) and all([isinstance(x, int) for x in filelist]):
        for files in filelist:
            try:
                im = Image(files, channel)
                im.save(lines[:-5]+'_'+channel+'.fits')
            except:
                print('Bad channel designator')

    else:
        print("Not valid name or file name list")
