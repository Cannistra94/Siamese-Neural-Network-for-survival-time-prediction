import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from numpy import random as rng
import random 
from sklearn.utils import resample
import pickle
import time

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, Activation, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from keras.optimizers import Adam, SGD
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
import scipy.io as sio

def loader(img, fill_nan=-1000):
    img = np.array(img).astype(float)

    if np.isnan(img).any():
        mask = np.isnan(img)
        img[mask] = fill_nan
    assert ~np.isnan(img).any()

    if img.ndim < 3:
        img = img[:, :, np.newaxis]

    return img

def rescale(data, maxi, mini):
    data_0_1 = [(x - mini) / (maxi - mini) for x in data]
    return np.asarray(data_0_1, dtype='float')

import training set and datapre-processing
import mat73


class1 = 'tr_same1.mat'
class0= 'tr_diff1.mat'

pairs1 =  mat73.loadmat(class1)


pairs0 =  mat73.loadmat(class0)

#training set
#similar couples labeled as 0
pairs_tr = [np.zeros((((len(pairs1["tr_same1"]['Immagine_1']))+(len(pairs0["tr_diff1"]['Immagine_1']))), 64, 64, 3)) for i in range(2)]
pairs_same_id=[]
pairs_same_label =[]

for idx in range (len(pairs1["tr_same1"]['Immagine_1'])):
    pairs_tr[0][idx, :, :, :] = loader((pairs1["tr_same1"]['Immagine_1'][idx]))
    pairs_tr[1][idx, :, :, :] = loader((pairs1["tr_same1"]['Immagine_2'][idx]))
    pairs_same_id+=[[(pairs1["tr_same1"]['IDPaziente_1'],pairs1["tr_same1"]['IDPaziente_2']) ]]  # patient_id for slice
    pairs_same_label.append(pairs1["tr_same1"]['Label'])  # labels # [4] for Adaptive
    

pairs_tr=np.array(pairs_tr,dtype=float)
pairs_same_id=np.array(pairs_same_id)
pairs_same_label = np.array(pairs_same_label)


#different couples labeled as 1
pairs_diff_id=[]
pairs_diff_label = []

n=(len(pairs1["tr_same1"]['Immagine_1']))-1;
for idx1 in range (len(pairs0["tr_diff1"]['Immagine_1'])):
    pairs_tr[0][n+idx1, :, :, :] = loader(pairs0["tr_diff1"]['Immagine_1'][idx1])
    pairs_tr[1][n+idx1,:, :, :] = loader(pairs0["tr_diff1"]['Immagine_2'][idx1])
    pairs_diff_id+=[[(pairs0["tr_diff1"]['IDPaziente_1'],pairs0["tr_diff1"]['IDPaziente_2']) ]] # patient_id for slice
    pairs_diff_label.append(pairs0["tr_diff1"]['Label']) # labels # [4] for Adaptive
    
    
pairs_tr=np.array(pairs_tr,dtype=float)
pairs_diff_id=np.array(pairs_diff_id)
pairs_diff_label = np.array(pairs_diff_label)

#Pre processing

maxi, mini = np.max(pairs_tr), np.min(pairs_tr)
pairs_tr = rescale(np.asarray(pairs_tr), maxi, mini)
pairs_tr=np.asarray(pairs_tr).astype(np.float32)

labels_tr=np.zeros(len(pairs_diff_label)+len(pairs_same_label))
for i in range(len(pairs_same_label)):
    labels_tr[i]=1
    
labels_tr=np.asarray(labels_tr).astype(np.float32)

image_shape=(64,64,3)
batch_size=64

pairs_tr,labels_tr=shuffle(pairs_tr,labels_tr)

x_train_1 = pairs_tr[0,:]  
x_train_2 = pairs_tr[1,:]</syntaxhighlight>

import validation set and datapre-processing 

class1 = 'siamese_dataset/val_same1.mat'
class0='siamese_dataset/val_diff1.mat'

load = sio.loadmat(class1)
pairs1 = load['val_same1'][0]

load = sio.loadmat(class0)
pairs0 = load['val_diff1'][0]

#coppie uguali
pairs_val = [np.zeros(( (pairs0.shape[0]+pairs1.shape[0]), 64, 64, 3)) for i in range(2)]
pairs_same_id_val=[]
pairs_same_label_val =[]

for idx in range(pairs1.shape[0]):
    pairs_val[0][idx, :, :, :] = loader(pairs1[idx][2])
    pairs_val[1][idx, :, :, :] = loader(pairs1[idx][3])
    pairs_same_id_val+=[[(pairs1[idx][0],pairs1[idx][1]) ]]  # patient_id for slice
    pairs_same_label_val.append(pairs1[idx][4][0])  # labels # [4] for Adaptive
    

pairs_val=np.array(pairs_val,dtype=float)
pairs_same_id_val=np.array(pairs_same_id_val)
pairs_same_label_val = np.array(pairs_same_label_val)


#different couples
pairs_diff_id_val=[]
pairs_diff_label_val = []

n=pairs1.shape[0]-1;
for idx1 in range(pairs0.shape[0]):
    pairs_val[0][n+idx1, :, :, :] = loader(pairs0[idx1][2])
    pairs_val[1][n+idx1,:, :, :] = loader(pairs0[idx1][3])
    pairs_diff_id_val+=[[(pairs0[idx1][0],pairs0[idx1][1])]]# patient_id for slice
    pairs_diff_label_val.append(pairs0[idx1][4][0])  # labels # [4] for Adaptive
    
    
pairs_val=np.array(pairs_val,dtype=float)
pairs_diff_id_val=np.array(pairs_diff_id_val)
pairs_diff_label_val = np.array(pairs_diff_label_val)

#Pre processing
#pairs=pairs.reshape(10000,2,64,64,3);
maxi, mini = np.max(pairs_val), np.min(pairs_val)
pairs_val = rescale(np.asarray(pairs_val), maxi, mini)
pairs_val=np.asarray(pairs_val).astype(np.float32)

labels_val=np.zeros(len(pairs_diff_label_val)+len(pairs_same_label_val))
for i in range(len(pairs_same_label_val)):
    labels_val[i]=1
    
labels_val=np.asarray(labels_val).astype(np.float32)

image_shape=(64,64,3)
batch_size=64

pairs_val,labels_val=shuffle(pairs_val,labels_val)

x_val_1 = pairs_val[0,:]  
x_val_2 = pairs_val[1,:]


pairs0 = load['test_diff1'][0]

#similar couples-TEST SET

pairs_test = [np.zeros(((pairs0.shape[0]+pairs1.shape[0]), 64, 64, 3)) for i in range(2)]
pairs_same_id_test=[]
pairs_same_label_test =[]

for idx in range(pairs1.shape[0]):
    pairs_test[0][idx, :, :, :] = loader(pairs1[idx][2])
    pairs_test[1][idx, :, :, :] = loader(pairs1[idx][3])
    pairs_same_id_test+=[[(pairs1[idx][0],pairs1[idx][1]) ]]  # patient_id for slice
    pairs_same_label_test.append(pairs1[idx][4][0])  # labels # [4] for Adaptive
    

pairs_test=np.array(pairs_test,dtype=float)
pairs_same_id_test=np.array(pairs_same_id_test)
pairs_same_label_test = np.array(pairs_same_label_test)


#different couples - TEST SET
pairs_diff_id_test=[]
pairs_diff_label_test = []

n=pairs1.shape[0]-1;
for idx1 in range(pairs0.shape[0]):
    pairs_test[0][n+idx1, :, :, :] = loader(pairs0[idx1][2])
    pairs_test[1][n+idx1,:, :, :] = loader(pairs0[idx1][3])
    pairs_diff_id_test+=[[(pairs0[idx1][0],pairs0[idx1][1])]]# patient_id for slice
    pairs_diff_label_test.append(pairs0[idx1][4][0])  # labels # [4] for Adaptive
    
    
pairs_test=np.array(pairs_test,dtype=float)
pairs_diff_id_test=np.array(pairs_diff_id_test)
pairs_diff_label_test = np.array(pairs_diff_label_test)

#Pre processing
#pairs=pairs.reshape(10000,2,64,64,3);
maxi, mini = np.max(pairs_test), np.min(pairs_test)
pairs_test = rescale(np.asarray(pairs_test), maxi, mini)
pairs_test=np.asarray(pairs_test).astype(np.float32)

labels_test=np.zeros(len(pairs_diff_label_test)+len(pairs_same_label_test))
for i in range(len(pairs_same_label_test)):
    labels_test[i]=1
    
labels_test=np.asarray(labels_test).astype(np.float32)

image_shape=(64,64,3)
batch_size=64

pairs_test,labels_test=shuffle(pairs_test,labels_test)

x_test_1 = pairs_test[0,:]  
x_test_2 = pairs_test[1,:]
