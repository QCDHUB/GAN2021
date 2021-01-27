import numpy as np
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, ActivityRegularization, Lambda, Concatenate, Permute, Convolution1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from keras import regularizers
from functools import partial
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

Mp  = 0.93827
Mpi = 0.1395

gamma_min = 3.0011189
gamma_max = 3.8368106


gfeaturesmean = np.load('gfeaturesmean.npy')
gfeaturesstd = np.load('gfeaturesstd.npy')
gamma_hist = np.load('gamma_hist.npy')
ppipxymean = gfeaturesmean[[1, 2, 4, 5]]
ppipxystd = gfeaturesstd[[1, 2, 4, 5]]

def gamma_dist_rand(N):
    y = np.random.choice(np.arange(0, 100), size = N, p = gamma_hist)
    y = gamma_min + (y + np.random.uniform(0, 1, N))*(gamma_max - gamma_min)/100
    return(y)

def detector_filter(n):
    
    SAMPLE_SIZE = int(1.5*n)

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

    results = results_origin*gfeaturesstd+gfeaturesmean
    gamma = results[:, 0]
    px = results[:, 1]
    py = results[:, 2]
    pz = results[:, 3]
    pipx = results[:, 4]
    pipy = results[:, 5]
    pipz = results[:, 6]

    pimx = -px - pipx
    pimy = -py - pipy
    pimz = gamma - pz - pipz

    pe = np.sqrt(px*px+py*py+pz*pz+Mp*Mp)
    pipe = np.sqrt(pipx*pipx+pipy*pipy+pipz*pipz+Mpi**2)
    
    M2pi = (gamma+Mp-pe-pipe)*(gamma+Mp-pe-pipe) - pimx*pimx - pimy*pimy - pimz*pimz
    M2pi = M2pi.reshape(-1, 1)
    
    filter = np.concatenate([theta, M2pi], axis=1)

    index = np.where((filter[:, 0]<=50)&(filter[:, 1]<=50)&(filter[:, 2]<=0.2))

    noise = noise[index]
    results_origin = results_origin[index]
    return results_origin[:n]

def make_generator():
    gamma = Input(shape=(1,))
    noise = Input(shape=(100,))
    visible = concatenate([gamma, noise])
    hidden1 = Dense(512)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    hidden2 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    hidden3 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden3)
    hidden4 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden4)
    hidden5 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    output = Dense(6)(LR)
    output2 = concatenate([gamma, output])

    generator = Model(inputs=[gamma, noise], outputs=[output2])
    return(generator)

generator = make_generator()
generator.load_weights("generatorfilter00224.h5")

def gen_samples(SAMPLE_SIZE = 1000):
    results = detector_filter(SAMPLE_SIZE)
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
    pimz = gamma - pz - pipz
    
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
    pime = np.sqrt(pimxsq+pimysq+pimzsq+Mpi**2)
    
    M2pi = (gamma+Mp-pe-pipe)*(gamma+Mp-pe-pipe) - pimxsq - pimysq - pimzsq

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
