#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The script takes as input the orginal clinical data in csv format and the cut_off in days 
the output is a csv file with two columns, the first the patients name, the second the new patients' labels 
"""

import pandas as pd
import numpy as np
import os
import math

###input data
input_path = '/data/NSCLC_dataset/database/NSCLC_Radiomics.csv'
output_path = '/data/NSCLC_dataset/database/'


cut_off_months =  12# months


#define filename of output csv
outputfilename = 'NSCLC_Radiomics_newlabel_cutoff_'+str(cut_off_months)+'months.csv'

#read input csv
clinicaldata = pd.read_csv(input_path)

#convert cut_off in days
cut_off = math.ceil(cut_off_months/2)*30+math.floor(cut_off_months/2)*31

outputdata = pd.DataFrame(columns = ['PatientID', 'deadstatus.event_new'])

#range over input csv and generate output csv
for index in range(len(clinicaldata)):
    if clinicaldata.loc[index,'deadstatus.event']==1:
        if clinicaldata.loc[index,'Survival.time'] < cut_off:
            outputdata.loc[index]= [clinicaldata.loc[index,'PatientID'], 1] 
        elif  clinicaldata.loc[index,'Survival.time'] >= cut_off:
            outputdata.loc[index]= [clinicaldata.loc[index,'PatientID'], 0] 
        
    elif clinicaldata.loc[index,'deadstatus.event']==0:
        if clinicaldata.loc[index,'Survival.time'] >= cut_off: 
            outputdata.loc[index]= [clinicaldata.loc[index,'PatientID'], 0] 
                
outputdata.to_csv(os.path.join(output_path, outputfilename))
