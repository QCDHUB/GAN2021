import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Input, 
                                     Dense, 
                                     Reshape, 
                                     Flatten, 
                                     Dropout, 
                                     ActivityRegularization, 
                                     Lambda, 
                                     Concatenate, 
                                     Permute, 
                                     Convolution1D, 
                                     MaxPooling1D, 
                                     AveragePooling1D, 
                                     GlobalAveragePooling1D,
                                     concatenate, 
                                     dot,
                                     BatchNormalization,
                                     LeakyReLU)
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcol
from sys import argv, exit
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
Mp  = 0.93827
Mpi = 0.1395
gamma_min = 1.1034470395531724
gamma_max = 1.2664411378747609
gfeaturesmean = np.load('gfeaturesmean.npy')
gfeaturesstd = np.load('gfeaturesstd.npy')
gamma_hist = np.load('gamma_hist.npy')
pimmean=gfeaturesmean[7:10]
pimstd=gfeaturesstd[7:10]
emean=gfeaturesmean[10:13]
estd=gfeaturesstd[10:13]
dotmean=gfeaturesmean[13:]
dotstd=gfeaturesstd[13:]
ppipxymean = gfeaturesmean[[1, 2, 4, 5]]
ppipxystd = gfeaturesstd[[1, 2, 4, 5]]

def gamma_dist_rand(N):
    y = np.random.choice(np.arange(0, 10000), size = N, p = gamma_hist)
    y = gamma_min + (y + np.random.uniform(0, 1, N))*(gamma_max - gamma_min)/10000
    return(y)

def feature_mul(x):
    featuresmean = K.constant(gfeaturesmean[:7])
    featuresstd = K.constant(gfeaturesstd[:7])
    xyz = x*featuresstd+featuresmean
    gamma = xyz[:, 0:1]
    px = xyz[:, 1:2]
    py = xyz[:, 2:3]
    pz = xyz[:, 3:4]
    pipx = xyz[:, 4:5]
    pipy = xyz[:, 5:6]
    pipz = xyz[:, 6:7]
    pimx = -px - pipx
    pimy = -py - pipy
    pimz = -pz - pipz
    pxsq = px*px
    pysq = py*py
    pzsq = pz*pz
    pipxsq = pipx*pipx
    pipysq = pipy*pipy
    pipzsq = pipz*pipz
    pimxsq = pimx*pimx
    pimysq = pimy*pimy
    pimzsq = pimz*pimz
    pe = K.sqrt(pxsq+pysq+pzsq+Mp*Mp)
    pipe = K.sqrt(pipxsq+pipysq+pipzsq+Mpi**2)
    pime = gamma + K.sqrt(Mp*Mp + gamma*gamma) - pe - pipe
    M2pi = pime*pime - pimxsq - pimysq - pimzsq
    p_pip = pe*pipe - px*pipx - py*pipy - pz*pipz
    p_pim = pe*pime - px*pimx - py*pimy - pz*pimz
    pip_pim = pipe*pime - pipx*pimx - pipy*pimy -pipz*pimz
    protone = K.sqrt(Mp*Mp + gamma*gamma)
    MMP = (gamma + protone - pe)*(gamma + protone - pe) - px*px - py*py - pz*pz
    m2pimmp = M2pi*MMP
    pim3 = K.concatenate([pimx, pimy, pimz])
    pim3 = (pim3 - K.constant(pimmean))/K.constant(pimstd)
    e = K.concatenate([pe, pipe, pime])
    e = (e - K.constant(emean))/K.constant(estd)
    dot = K.concatenate([M2pi, p_pip, p_pim, pip_pim, MMP, m2pimmp])
    dot = (dot - K.constant(dotmean))/K.constant(dotstd)
    return K.concatenate([pim3, e, dot])

def make_generator():
    gamma = Input(shape=(1,))
    noise = Input(shape=(100,))
    visible = concatenate([gamma, noise])
    hidden1 = Dense(256)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    BN = BatchNormalization(momentum=0.8)(LR)
    hidden2 = Dense(512)(BN)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    BN = BatchNormalization(momentum=0.8)(LR)
    hidden3 = Dense(1024)(BN)
    LR = LeakyReLU(alpha=0.2)(hidden3)
    output = Dense(6)(LR)
    output2 = concatenate([gamma, output])
    features = Lambda(feature_mul)(output2)
    outputmerge = concatenate([output2, features])
    generator = Model(inputs=[gamma, noise], outputs=[outputmerge])
    return(generator)

generator = make_generator()
generator.load_weights("generatorfilter00466.h5")

def detector_filter(n):
    SAMPLE_SIZE = int(1.4*n)
    noise = np.random.normal(0, 1, [SAMPLE_SIZE, 100])
    gamma = gamma_dist_rand(SAMPLE_SIZE)
    gamma = (gamma - gfeaturesmean[0])/gfeaturesstd[0]
    results_origin = generator.predict([gamma, noise])
    ppipxy = results_origin[:, [1, 2, 4, 5]]
    ppipxy = ppipxy*ppipxystd+ppipxymean
    ptheta = np.arctan2(ppipxy[:, 1], ppipxy[:, 0])
    ptheta = ptheta.reshape(-1, 1)
    piptheta = np.arctan2(ppipxy[:, 3], ppipxy[:, 2])
    piptheta = piptheta.reshape(-1, 1)
    theta = np.concatenate([ptheta, piptheta], axis=1)
    theta = np.floor(theta/np.pi*180+205)%60
    index = np.where((theta[:, 0]<=50)&(theta[:, 1]<=50))
    noise = noise[index]
    results_origin = results_origin[index]
    gamma = gamma[index]
    return gamma[:n], noise[:n], results_origin[:n]

def gen_samples(SAMPLE_SIZE = 1000):
    gamma, noise, results = detector_filter(SAMPLE_SIZE)
    results = results*gfeaturesstd+gfeaturesmean
    gamma = results[:, 0]
    px    = results[:, 1]
    py    = results[:, 2]
    pz    = results[:, 3]
    pipx  = results[:, 4]
    pipy  = results[:, 5]
    pipz  = results[:, 6]
    pimx = -px - pipx
    pimy = -py - pipy
    pimz = - pz - pipz
    pxsq = px*px
    pysq = py*py
    pzsq = pz*pz
    pipxsq = pipx*pipx
    pipysq = pipy*pipy
    pipzsq = pipz*pipz
    pimxsq = pimx*pimx
    pimysq = pimy*pimy
    pimzsq = pimz*pimz
    pe = np.sqrt(pxsq+pysq+pzsq+Mp**2)
    pipe = np.sqrt(pipxsq+pipysq+pipzsq+Mpi**2)
    pime = gamma + np.sqrt(Mp**2 + gamma*gamma) - pe - pipe   
    M2pi = pime*pime - pimxsq - pimysq - pimzsq
    return np.stack([ M2pi, 
                      gamma, 
                      px, 
                      pipx, 
                      pimx, 
                      py, 
                      pipy, 
                      pimy, 
                      pz, 
                      pipz, 
                      pimz, 
                      pe, 
                      pipe, 
                      pime], axis=1)
