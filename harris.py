# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:43:10 2021

@author: Dell
"""


import numpy as np
from skimage.feature import corner_harris, peak_local_max,corner_peaks
import cv2
import matplotlib.pyplot as plt


def get_harris_corners(im, edge_discard=15):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 15

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = corner_peaks(h, min_distance=1, indices=True)
    print(coords.shape)
    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert(dimx == dimc, 'Data dimension does not match dimension of centers')

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)
"""
img1 = cv2.imread("F:\ML\Image_Color\im_l.png")
img2 = cv2.imread("F:\ML\Image_Color\im_r.png")

### PART 1

#Converting images to gray-scale

print(img1.shape,img2.shape)
grayImg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayImg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

h, curr_coords = get_harris_corners(grayImg1)
print(h.shape)
print("curr_coords",curr_coords.shape)
implot = plt.imshow(img1)
# Red dots of size 40
plt.scatter(x=curr_coords[1], y=curr_coords[0], c='r', s=10)
plt.show()


h1, curr_coords1 = get_harris_corners(grayImg2)

#print("curr_coords",curr_coords.shape)
implot = plt.imshow(img2)
# Red dots of size 40
plt.scatter(x=curr_coords1[1], y=curr_coords1[0], c='r', s=10)
plt.show()
"""