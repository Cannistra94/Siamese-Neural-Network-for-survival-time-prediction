# -*- coding: utf-8 -*-
"""

this script takes as input the path of the files, and two lists build as file "examplebenign " 
the script uses both lists to organize all images of the current patient into two subfolders. 

"""

from glob import glob
import os
import shutil



data_path ='.../data/NSCLC_Radiomics'# folder with original files
    
path_listbenign = '/.../listbenign'
path_listmalignant = '/.../listmalignant'

outputpath_benignpatients = '.../benign/'
outputpath_malignantpatients = '.../malignant/'
###

#create output folder if not exists
if not os.path.exists(outputpath_benignpatients):
    os.makedirs(outputpath_benignpatients)
if not os.path.exists(outputpath_malignantpatients):
    os.makedirs(outputpath_malignantpatients)

# Crea la lista di tutti i file contenuti
patient_list = glob(data_path + "/*.nrrd")

##alive
listalive=[]
with open(path_listbenign) as file:
    for line in file:
        line = line.strip() #preprocess line
        listalive.append(line)
    
for patient in patient_list:
    # Estrazione del nome del paziente:
    stringa = os.path.basename(patient).rsplit('_')[0]
    
    if stringa in listalive:
        patient_code = stringa + os.path.basename(patient.replace(stringa, '').split('.',1)[0])+'.nrrd'
        shutil.copyfile(data_path+patient_code, outputpath_benignpatients+patient_code)

##dead
listdead=[]
with open( path_listmalignant) as file:
    for line in file:
        line = line.strip() #preprocess line
        listdead.append(line)
    
for patient in patient_list:
    # Estrazione del nome del paziente:
    stringa = os.path.basename(patient).rsplit('_')[0]
    
    if stringa in listdead:
        patient_code = stringa + os.path.basename(patient.replace(stringa, '').split('.',1)[0])+'.nrrd'
        shutil.copyfile(data_path+patient_code, outputpath_malignantpatients+patient_code)
