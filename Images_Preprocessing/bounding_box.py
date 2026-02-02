#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The script takes as input the path of the nrrd images and mask.
processing all patients the script loads the nrrd images and segmentation for 
each patient (imagedata, maskdata) and findes the maximum and minimum index for each dimension. These indices 
were used to crop the original image and keep only the segmented area and the surronding contained in the bounding box
for each patient a new file is saved is the selected outpath
"""

import numpy as np
import nrrd
import os


### input data path#
path= '/.../02.Images_mask_cropped/images+mask/'
outpath = '.../NSCLC_ImageSegBoundingBox'
###

for m in range(1,422+1):
    if m == 128:
        continue
    patientimage = 'LUNG1-'+str(m).zfill(3)+'_image.nrrd'
    patientmask = 'LUNG1-'+str(m).zfill(3)+'_mask.nrrd'
    imagepath = os.path.join(path, patientimage)
    maskpath = os.path.join(path, patientmask)

    
    imagedata, imageheader = nrrd.read(imagepath)
    maskdata, maskheader = nrrd.read(maskpath)
    
    
    i, j, k = np.where(maskdata)
    shapek= maskdata.shape[2]
    
    indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                          np.arange(min(j), max(j) + 1),
                          np.arange(shapek- max(k) + 1, shapek-min(k), ),
                          indexing='xy')
    sub_image = imagedata[tuple(indices)]
    
    filename = 'LUNG1-'+str(m).zfill(3)+'_croppedimage.nrrd'
    outpathfile = os.path.join(outpath,filename)
    nrrd.write(outpathfile, sub_image)
