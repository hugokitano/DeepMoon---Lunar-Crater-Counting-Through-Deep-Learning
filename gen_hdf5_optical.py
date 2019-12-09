#!/usr/bin/env python
"""Input Image Dataset Generator

Script for generating input datasets from Lunar global digital elevation maps 
(DEMs) and crater catalogs.

This script is designed to use the LRO-Kaguya DEM and a combination of the
LOLA-LROC 5 - 20 km and Head et al. 2010 >=20 km crater catalogs.  It
generates a randomized set of small (projection-corrected) images and
corresponding crater targets.  The input and target image sets are stored as
hdf5 files.  The longitude and latitude limits of each image is included in the
input set file, and tables of the craters in each image are stored in a
separate Pandas HDFStore hdf5 file.

The script's parameters are located under the Global Variables.  We recommend
making a copy of this script when generating a dataset.

"""

########## Imports ##########

# Python 2.7 compatibility.
from __future__ import absolute_import, division, print_function

from PIL import Image
import input_data_gen as igen
import time

import cv2
import h5py

import numpy as np
import pandas as pd

########## Global Variables ##########

# dataset : should be one of train, valid or test
dataset = 'train'

# Output filepath and file header.  Eg. if outhead = "./input_data/train",
# files will have extension "./out/train_inputs.hdf5" and
# "./out/train_targets.hdf5"
outhead = "./input_data/" + dataset


# Range of image widths, in pixels, to crop from source image (input images
# will be scaled down to ilen). For Orthogonal projection, larger images are
# distorted at their edges, so there is some trade-off between ensuring images
# have minimal distortion, and including the largest craters in the image.
rawlen_range = [500, 6500]

# Distribution to sample from rawlen_range - "uniform" for uniform, and "log"
# for loguniform.
rawlen_dist = 'log'

# Size of input images.
ilen = 256

# Size of target images.
tglen = 256

# Minimum pixel diameter of craters to include in in the target.
minpix = 1.


### Target mask arguments. ###

# If True, truncate mask where image has padding.
truncate = True

# If rings = True, thickness of ring in pixels.
ringwidth = 1

# If True, script prints out the image it's currently working on.
verbose = True


########## Script ##########

def _read_names(names_file):
    ''' Read the names.txt file and return a list of all bags '''
    with open(names_file) as f:
        names = f.read().splitlines()
    return names

def circlemaker(r=10.):
    """
    Creates circle mask of radius r.
    """
    # Based on <https://stackoverflow.com/questions/10031580/
    # how-to-write-simple-geometric-shapes-into-numpy-arrays>

    # Mask grid extent (+1 to ensure we capture radius).
    rhext = int(r) + 1

    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]
    circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)


def ringmaker(r=10., dr=1):
    """
    Creates ring of radius r and thickness dr.
    Parameters
    ----------
    r : float
        Ring radius
    dr : int
        Ring thickness (cv2.circle requires int)
    """
    # See <http://docs.opencv.org/2.4/modules/core/doc/
    # drawing_functions.html#circle>, supplemented by
    # <http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html>
    # and <https://github.com/opencv/opencv/blob/
    # 05b15943d6a42c99e5f921b7dbaa8323f3c042c6/modules/imgproc/
    # src/drawing.cpp>.

    # mask grid extent (dr/2 +1 to ensure we capture ring width
    # and radius); same philosophy as above
    rhext = int(np.ceil(r + dr / 2.)) + 1

    # cv2.circle requires integer radius
    mask = np.zeros([2 * rhext + 1, 2 * rhext + 1], np.uint8)

    # Generate ring
    ring = cv2.circle(mask, (rhext, rhext), int(np.round(r)), 1, thickness=dr)

    return ring.astype(float)



def get_merge_indices(cen, imglen, ks_h, ker_shp):
    """Helper function that returns indices for merging stencil with base
    image, including edge case handling.  x and y are identical, so code is
    axis-neutral.
    Assumes INTEGER values for all inputs!
    """

    left = cen - ks_h
    right = cen + ks_h + 1

    # Handle edge cases.  If left side of stencil is beyond the left end of
    # the image, for example, crop stencil and shift image index to lefthand
    # side.
    if left < 0:
        img_l = 0
        g_l = -left
    else:
        img_l = left
        g_l = 0
    if right > imglen:
        img_r = imglen
        g_r = ker_shp - (right - imglen)
    else:
        img_r = right
        g_r = ker_shp

    return [img_l, img_r, g_l, g_r]


def make_mask(craters, img, binary=True, rings=False, ringwidth=1,
              truncate=True):
    """Makes crater mask binary image (does not yet consider crater overlap).

    Parameters
    ----------
    craters : pandas.DataFrame
        Craters catalogue that includes pixel x and y columns.
    img : numpy.ndarray
        Original image; assumes colour channel is last axis (tf standard).
    binary : bool, optional
        If True, returns a binary image of crater masks.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.

    Returns
    -------
    mask : numpy.ndarray
        Target mask image.
    """

    # Load blank density map
    imgshape = img.shape[:2]
    mask = np.zeros(imgshape)
    cx = (craters["x1"].values.astype('int') + craters["x2"].values.astype('int')) / 2
    cy = (craters["y1"].values.astype('int') + craters["y2"].values.astype('int')) / 2 
    x_rad = abs(craters["x1"].values.astype('int') - craters["x2"].values.astype('int')) / 2
    y_rad = abs(craters["y1"].values.astype('int') - craters["y2"].values.astype('int')) / 2
    radius = (x_rad + y_rad) / 2.
    
    #print ('craters : ', craters.head)
    
    craters["x"] = cx   # add these new derived columns
    craters["y"] = cy
    craters["radius"] = radius
    
    #print ('craters_new: ', craters.head)

    for i in range(craters.shape[0]):
        if rings:
            kernel = ringmaker(r=radius[i], dr=ringwidth)
        else:
            kernel = circlemaker(r=radius[i])
        # "Dummy values" so we can use get_merge_indices
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2

        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(int(cx[i]), imgshape[1],
                                                   ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(int(cy[i]), imgshape[0],
                                                   ks_half, kernel_support)

        # Add kernel to image
        mask[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

    if binary:
        mask = (mask > 0).astype(float)

    if truncate:
        if img.ndim == 3:
            mask[img[:, :, 0] == 0] = 0
        else:
            mask[img == 0] = 0

    return mask


def genDataset(outhead, rawlen_range=[1000, 2000],
               rawlen_dist='log', ilen=ilen,
               minpix=0, tglen=tglen, binary=True, rings=True,
               ringwidth=1, truncate=True, istart=0, seed=None,
               verbose=False):
    """Generates random dataset from a global DEM and crater catalogue.

    The function randomly samples small images from a global digital elevation
    map (DEM) that uses a Plate Carree projection, and converts the small
    images to Orthographic projection.  Pixel coordinates and radii of craters
    from the catalogue that fall within each image are placed in a
    corresponding Pandas dataframe.  Images and dataframes are saved to disk in
    hdf5 format.

    Parameters
    ----------
    outhead : str
        Filepath and file prefix of the image and crater table hdf5 files.
    rawlen_range : list-like, optional
        Lower and upper bounds of raw image widths, in pixels, to crop from
        source.  To always crop the same sized image, set lower bound to the
        same value as the upper.  Default is [300, 4000].
    rawlen_dist : 'uniform' or 'log'
        Distribution from which to randomly sample image widths.  'uniform' is
        uniform sampling, and 'log' is loguniform sampling.
    ilen : int, optional
        Input image width, in pixels.  Cropped images will be downsampled to
        this size.  Default is 256.
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        LRO-Kaguya's [-180., 180., -60., 60.].
    arad : float. optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimum crater diameter in pixels to be included in crater list.
        Useful when the smallest craters in the catalogue are smaller than 1
        pixel in diameter.
    tglen : int, optional
        Target image width, in pixels. Default is 256
    binary : bool, optional
        If True, returns a binary image of crater masks.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.
    istart : int
        Output file starting number, when creating datasets spanning multiple
        files.
    seed : int or None
        np.random.seed input (for testing purposes).
    verbose : bool
        If True, prints out number of image being generated.
    """

    # just in case we ever make this user-selectable...
    origin = "upper"

    # Seed random number generator.
    np.random.seed(seed)

    # Get craters.
    #AddPlateCarree_XY(craters, list(img.size), cdim=cdim, origin=origin)

    #iglobe = ccrs.Globe(semimajor_axis=arad*1000., semiminor_axis=arad*1000., ellipse=None)

    # Create random sampler (either uniform or loguniform).
    if rawlen_dist == 'log':
        rawlen_min = np.log10(rawlen_range[0])
        rawlen_max = np.log10(rawlen_range[1])

        def random_sampler():
            return int(10**np.random.uniform(rawlen_min, rawlen_max))
    else:

        def random_sampler():
            return np.random.randint(rawlen_range[0], rawlen_range[1] + 1)

        

    filenames = _read_names('LRO-equator/Names/' + dataset + '.txt')

    # number of images to produce
    amt = len(filenames)

    # Initialize output hdf5s.
    imgs_h5 = h5py.File(outhead + '_images.hdf5', 'w')
    imgs_h5_inputs = imgs_h5.create_dataset("input_images", (amt, ilen, ilen),
                                            dtype='uint8')
    imgs_h5_inputs.attrs['definition'] = "Input image dataset."
    imgs_h5_tgts = imgs_h5.create_dataset("target_masks", (amt, tglen, tglen),
                                          dtype='float32')
    imgs_h5_tgts.attrs['definition'] = "Target mask dataset."
    imgs_h5_llbd = imgs_h5.create_group("longlat_bounds")
    imgs_h5_llbd.attrs['definition'] = ("(long min, long max, lat min, "
                                        "lat max) of the cropped image.")
    imgs_h5_box = imgs_h5.create_group("pix_bounds")
    imgs_h5_box.attrs['definition'] = ("Pixel bounds of the Global DEM region"
                                       " that was cropped for the image.")
    imgs_h5_dc = imgs_h5.create_group("pix_distortion_coefficient")
    imgs_h5_dc.attrs['definition'] = ("Distortion coefficient due to "
                                      "projection transformation.")
    imgs_h5_cll = imgs_h5.create_group("cll_xy")
    imgs_h5_cll.attrs['definition'] = ("(x, y) pixel coordinates of the "
                                       "central long / lat.")
    craters_h5 = pd.HDFStore(outhead + '_craters.hdf5', 'w')

    # Zero-padding for hdf5 keys.
    zeropad = int(np.log10(amt)) + 1

    for i, name in enumerate(filenames):
        img = Image.open('LRO-equator/Images/' + name + '.jpg').convert("L")
        width, height = img.size
        #print ('shape of image', img.size)
        im = img.resize([ilen, ilen], resample=Image.NEAREST)
        #print ('shape of image', im.size)
        
        imgo_arr = np.asanyarray(im)
        assert imgo_arr.sum() > 0, ("Sum of imgo is zero!  There likely was "
                                    "an error in projecting the cropped "
                                    "image.")
        
        
        craters = pd.read_csv('LRO-equator/Annotations_YOLO/' + name + '.txt',  delimiter= '\s+',
                          names=['x1', 'y1', 'x2', 'y2'])
        
        #print('craters before scaling : ')
        #print (craters)
        
        craters['x1'] = craters['x1'].apply(lambda x: x * ((ilen + 0.0) / width))
        craters['x2'] = craters['x2'].apply(lambda x: x * ((ilen + 0.0) / width))
        craters['y1'] = craters['y1'].apply(lambda x: x * ((ilen + 0.0) / height))
        craters['y2'] = craters['y2'].apply(lambda x: x * ((ilen + 0.0) / height))
        
        if (craters.max().any() > ilen):
            print ('####################\n\n oh noooooooooooooooo', craters.max)
        
        #print('craters after scaling : ')
        #print (craters)
        
        tgt = np.asanyarray(img.resize((tglen, tglen),
                                        resample=Image.BILINEAR))
        mask = make_mask(craters, tgt, binary=binary, rings=rings,
                         ringwidth=ringwidth, truncate=truncate)

        # Output everything to file.
        imgs_h5_inputs[i, ...] = imgo_arr
        imgs_h5_tgts[i, ...] = mask
        
        #print ('craters head after adding x, y and radius')
        #print (craters.head())
        
        craters_h5[name] = craters
        imgs_h5.flush()
        craters_h5.flush()

    imgs_h5.close()
    craters_h5.close()

if __name__ == '__main__':

    start_time = time.time()

    # Utilize mpi4py for multithreaded processing.
    istart = 0

    # Generate input images.
    genDataset(outhead=outhead, rawlen_range=rawlen_range,
                    rawlen_dist=rawlen_dist, ilen=ilen,
                    minpix=minpix, tglen=tglen, binary=True,
                    rings=True, ringwidth=ringwidth, truncate=truncate,
                    istart=istart, verbose=verbose)

    elapsed_time = time.time() - start_time
    if verbose:
        print("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))

