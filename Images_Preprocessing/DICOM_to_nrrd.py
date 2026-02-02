#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This script goes through all folders in 'NSCLC-Radiomics' of patients 
#and finds the segmentation and the images paths which are saved in lstFilesDCM


import os
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
import numpy as np

reader = sitk.ImageSeriesReader()


###INPUT DATA --to be modified##
globalpath = '/.../01.Raw_data/NSCLC-Radiomics'
pathout = '/.../02.Images_mask_cropped/images+mask'
###

patientlist = os.listdir(globalpath)


for j in range(1,len(patientlist)+1):
    patient = 'LUNG1-'+str(j).zfill(3)
    patientpath = os.path.join(globalpath,patient)
    imagepath = 0
    
    #find segmentation and image path for each patient
    for r, d, f in os.walk(patientpath):
        for file in d:
            if 'Segmentation' in file:
                segmentationpath = os.path.join(r,file)
                
                for root, directory, files in os.walk(segmentationpath):
                    segmentationpath = os.path.join(segmentationpath,files[0])
            
            if 'StudyID' in file: 
                imagepath = os.path.join(r,file)
    if imagepath == 0.:
        print('error')
        break
    
    #choose filename as save segmentation to nrrd in selected folder
    ds = pydicom.read_file(segmentationpath, force=True)
    pathimagemask = os.path.join(pathout, patient+'_mask.nrrd' )
    ds.save_as(pathimagemask)
    
    #select dirname of original dcm image
    dirName = os.path.join(imagepath, os.listdir(imagepath)[0])
    
    dicom_names = reader.GetGDCMSeriesFileNames(dirName)
    dicom_names_flip= np.flipud(dicom_names) 
    
    reader.SetFileNames(dicom_names_flip)
    image = reader.Execute()
   
    pathimage = os.path.join(pathout, patient+'_image.nrrd' )
    sitk.WriteImage(image,pathimage)
