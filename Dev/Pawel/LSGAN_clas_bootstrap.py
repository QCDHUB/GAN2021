#!/usr/bin/env python
# coding: utf-8

# LS-GAN for CLAS data
# 
# I used LS-GAN to replace WGAN - yielding better matching
# Still not perfect matching in Phi angles yet
# 
# Yaohang Li 2/28/2021



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

try:
    runno = int( argv[1] )
except:
    print( '\nWARNING: arguments are missing' )
    print( 'USAGE: python3 %s <runno>' % ( argv[0].split('/')[-1] ) )
    exit()


tf.config.threading.set_inter_op_parallelism_threads( 16 )


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
imagedir = './'



def gamma_dist_rand(N):
    y = np.random.choice(np.arange(0, 10000), size = N, p = gamma_hist)
    y = gamma_min + (y + np.random.uniform(0, 1, N))*(gamma_max - gamma_min)/10000
    
    return(y)



def detector_filter(n):
    
    SAMPLE_SIZE = int(1.40*n)

    noise = np.random.normal(0, 1, [SAMPLE_SIZE, 100])
    gamma = gamma_dist_rand(SAMPLE_SIZE)
    gamma = (gamma - gammamean)/gammastd
    results_origin = generator.predict([gamma, noise], batch_size=BATCH_SIZE)
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


pmass = 0.93827
pipmass = 0.1395
pimmass = 0.1395


# datapath = "../data/"
# events = X = np.empty(shape=[0, 14])
# for i in range(10):
#     temp = np.load(datapath+'evts-%d'%(i)+".npy")
#     events = np.concatenate([events, temp])
datapath = "/work/JAM/pawel/"
events = np.load(datapath+'evts-clas.npy')

px = events[:, 2]
py = events[:, 5]
pphi = np.arctan2(py, px)
pphi = pphi.reshape(-1, 1)

pipx = events[:, 3]
pipy = events[:, 6]
pipphi = np.arctan2(pipy, pipx)
pipphi = pipphi.reshape(-1, 1)

phi = np.concatenate([pphi, pipphi], axis=1)
phi = np.floor(phi/np.pi*180+205)%60

index = np.where((phi[:, 0]<=50)&(phi[:, 1]<=50))

events = events[index]
events.shape

M2pi = events[:, 0]
gamma = events[:, 1]
px = events[:, 2]
py = events[:, 5]
pz = events[:, 8]
pe = events[:, 11]
pipx = events[:, 3]
pipy = events[:, 6]
pipz = events[:, 9]
pipe = events[:, 12]
pimx = events[:, 4]
pimy = events[:, 7]
pimz = events[:, 10]
pime = events[:, 13]

v0 = gamma + pmass
v3 = gamma
v0prime = np.sqrt(v0*v0 - v3*v3)
c = v0*v0prime/(v0*v0 - v3*v3)
s = -c*v3/v0

cmgamma = c*gamma+s*gamma
cmprotone = c*pmass
cmprotonz = s*pmass
cmpe = c*pe + s*pz
cmpx = px
cmpy = py
cmpz = s*pe + c*pz
cmpipe = c*pipe + s*pipz
cmpipx = pipx
cmpipy = pipy
cmpipz = s*pipe + c*pipz
cmpime = cmgamma + cmprotone - cmpe - cmpipe
cmpimx = pimx
cmpimy = pimy
cmpimz = cmgamma + cmprotonz - cmpz - cmpipz

gamma = cmgamma
protone = cmprotone
protonz = cmprotonz
pe = cmpe
px = cmpx
py = cmpy
pz = cmpz
pipe = cmpipe
pipx = cmpipx
pipy = cmpipy
pipz = cmpipz
pime = cmpime
pimx = cmpimx
pimy = cmpimy
pimz = cmpimz

events = np.stack([M2pi, gamma, px, pipx, pimx, py, pipy, pimy, pz, pipz, pimz, pe, pipe, pime], axis=1)

p_pip = pe*pipe - px*pipx - py*pipy - pz*pipz
p_pim = pe*pime - px*pimx - py*pimy - pz*pimz
pip_pim = pipe*pime - pipx*pimx - pipy*pimy -pipz*pimz

MMP = (gamma + protone - pe)*(gamma + protone - pe) - px*px - py*py - (gamma + protonz - pz)*(gamma + protonz - pz)
m2pimmp = M2pi*MMP

gamma_dist = gamma
gamma_min = np.min(gamma_dist)
gamma_max = np.max(gamma_dist)
[gamma_hist, gamma_edges] = np.histogram(gamma_dist, bins = 10000, range=[gamma_min, gamma_max])

gamma_hist = gamma_hist/gamma_dist.shape[0]
np.save('gamma_hist.npy', gamma_hist)

clas_origin = np.stack([gamma, px, py, pz, pipx, pipy, pipz, pimx, pimy, pimz, pe, pipe, pime, M2pi, p_pip, p_pim, pip_pim, MMP, m2pimmp], axis = 1)

# normalization
clasmean = np.mean(clas_origin, axis = 0)
classtd = np.std(clas_origin, axis = 0)
clas = (clas_origin - clasmean)/classtd
gfeaturesmean = clasmean[0:7]
gfeaturesstd = classtd[0:7]
ppipxymean = clasmean[[1, 2, 4, 5]]
ppipxystd = classtd[[1, 2, 4, 5]]
gammamean = clasmean[0:1]
gammastd = classtd[0:1]
pmean = clasmean[1:4]
pstd = classtd[1:4]
pipmean = clasmean[4:7]
pipstd = classtd[4:7]
pimmean = clasmean[7:10]
pimstd = classtd[7:10]
emean = clasmean[10:13]
estd = classtd[10:13]
dotmean = clasmean[13:]
dotstd = classtd[13:]

np.save('gfeaturesmean.npy', gfeaturesmean)
np.save('gfeaturesstd.npy', gfeaturesstd)

index = np.where(events[:, 0]>0.08)
events_cut = events[index]
MMP_cut = MMP[index]


def feature_mul(x):
    featuresmean = K.constant(gfeaturesmean)
    featuresstd = K.constant(gfeaturesstd)
    
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
    
    pe = K.sqrt(pxsq+pysq+pzsq+pmass*pmass)
    pipe = K.sqrt(pipxsq+pipysq+pipzsq+pipmass*pipmass)
    pime = gamma + K.sqrt(pmass*pmass + gamma*gamma) - pe - pipe
    
    M2pi = pime*pime - pimxsq - pimysq - pimzsq
    p_pip = pe*pipe - px*pipx - py*pipy - pz*pipz
    p_pim = pe*pime - px*pimx - py*pimy - pz*pimz
    pip_pim = pipe*pime - pipx*pimx - pipy*pimy -pipz*pimz
    protone = K.sqrt(pmass*pmass + gamma*gamma)
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
    BN = BatchNormalization(momentum=0.8)(LR)
    output = Dense(6)(LR)
    output2 = concatenate([gamma, output])
    features = Lambda(feature_mul)(output2)
    
    outputmerge = concatenate([output2, features])

    generator = Model(inputs=[gamma, noise], outputs=[outputmerge])
    return(generator)


def make_discriminator():
    visible = Input(shape=(clas.shape[1],))
    hidden1 = Dense(512)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    DR = Dropout(rate=0.01)(LR)
    hidden2 = Dense(256)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    DR = Dropout(rate=0.01)(LR)
    
    output = Dense(1)(DR)

    discriminator = Model(inputs=[visible], outputs=output)
    return discriminator


optimizer = Adam(0.000001, 0.5)

# Build and compile the discriminator
discriminator = make_discriminator()
discriminator.compile(loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'])


generator = make_generator()

# The generator takes noise as input and generated imgs
generator_gamma = Input(shape=(1,))
generator_noise = Input(shape=(100,))
img = generator([generator_gamma, generator_noise])

# For the combined model we will only train the generator
discriminator.trainable = False

# The valid takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined = Model([generator_gamma, generator_noise], valid)
# (!!!) Optimize w.r.t. MSE loss instead of crossentropy
combined.compile(loss='mse', optimizer=optimizer)


BATCH_SIZE = 40000
TRAINING_RATIO = 1
valid = np.ones((BATCH_SIZE, 1))
fake = np.zeros((BATCH_SIZE, 1))


generator.load_weights("generatorfilter00466.h5")
discriminator.load_weights("discriminatorfilter00466.h5")


# for epoch in range(20400, 1000000):
for epoch in range(46600, 1000000):
       
    np.random.shuffle(clas)
    # print("Number of batches: ", int(dataset.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(clas.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = clas[i * minibatches_size:
                                            (i + 1) * minibatches_size]

        gamma, noise, fake_minibatches = detector_filter(minibatches_size)

        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:
                                                    (j + 1) * BATCH_SIZE]
            fake_batch = fake_minibatches[j * BATCH_SIZE:
                                                    (j + 1) * BATCH_SIZE]

            d_loss_real = discriminator.train_on_batch( image_batch, valid )
            d_loss_fake = discriminator.train_on_batch( fake_batch, fake )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss = d_loss_fake

        noise = noise[:BATCH_SIZE]
        gamma = gamma[:BATCH_SIZE]
        image_batch = discriminator_minibatches[:BATCH_SIZE]
        g_loss = combined.train_on_batch([gamma, noise], valid)

    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    
    if epoch%100==0:
        print(epoch)
        SAMPLE_SIZE = events.shape[0]
#         noise= np.random.normal(0, 1, [SAMPLE_SIZE, 100])
        gamma, noise, results = detector_filter(SAMPLE_SIZE)
        
        results = results*classtd+clasmean
        gamma = results[:, 0]
        px = results[:, 1]
        py = results[:, 2]
        pz = results[:, 3]
        pe = results[:, 10]
        pipx = results[:, 4]
        pipy = results[:, 5]
        pipz = results[:, 6]
        pipe = results[:, 11]
        pimx = results[:, 7]
        pimy = results[:, 8]
        pimz = results[:, 9]
        pime = results[:, 12]
        M2pi = pime*pime - pimx*pimx - pimy*pimy - pimz*pimz
        ganevents = np.stack([M2pi, gamma, px, pipx, pimx, py, pipy, pimy, pz, pipz, pimz, pe, pipe, pime], axis=1)

        ganpt = np.sqrt(ganevents[:, 2]*ganevents[:, 2] + ganevents[:, 5]*ganevents[:, 5])
        ganptheta = np.arctan2(ganpt, ganevents[:, 8])

        claspt = np.sqrt(events[:, 2]*events[:, 2] + events[:, 5]*events[:, 5])
        clasptheta = np.arctan2(claspt, events[:, 8])
        
        ganpipt = np.sqrt(ganevents[:, 3]*ganevents[:, 3] + ganevents[:, 6]*ganevents[:, 6])
        ganpiptheta = np.arctan2(ganpipt, ganevents[:, 9])

        claspipt = np.sqrt(events[:, 3]*events[:, 3] + events[:, 6]*events[:, 6])
        claspiptheta = np.arctan2(claspipt, events[:, 9])
        
        ganpimt = np.sqrt(ganevents[:, 4]*ganevents[:, 4] + ganevents[:, 7]*ganevents[:, 7])
        ganpimtheta = np.arctan2(ganpimt, ganevents[:, 10])

        claspimt = np.sqrt(events[:, 4]*events[:, 4] + events[:, 7]*events[:, 7])
        claspimtheta = np.arctan2(claspimt, events[:, 10])

        if 1:
            ganpt = np.sqrt(ganevents[:, 2]*ganevents[:, 2] + ganevents[:, 5]*ganevents[:, 5])
            ganptheta = np.arctan2(ganpt, ganevents[:, 8])
    
            claspt = np.sqrt(events[:, 2]*events[:, 2] + events[:, 5]*events[:, 5])
            clasptheta = np.arctan2(claspt, events[:, 8])
            
            ganpipt = np.sqrt(ganevents[:, 3]*ganevents[:, 3] + ganevents[:, 6]*ganevents[:, 6])
            ganpiptheta = np.arctan2(ganpipt, ganevents[:, 9])
    
            claspipt = np.sqrt(events[:, 3]*events[:, 3] + events[:, 6]*events[:, 6])
            claspiptheta = np.arctan2(claspipt, events[:, 9])
            
            ganpimt = np.sqrt(ganevents[:, 4]*ganevents[:, 4] + ganevents[:, 7]*ganevents[:, 7])
            ganpimtheta = np.arctan2(ganpimt, ganevents[:, 10])
    
            claspimt = np.sqrt(events[:, 4]*events[:, 4] + events[:, 7]*events[:, 7])
            claspimtheta = np.arctan2(claspimt, events[:, 10])

            counts, edges = np.histogram(ganptheta, bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins], axis=1 )
            np.save("lsgan-ptheta-%d-%d.npy" % (runno, epoch//100), readout)

            counts, edges = np.histogram(ganpiptheta, bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins], axis=1 )
            np.save("lsgan-piptheta-%d-%d.npy" % (runno, epoch//100), readout)

            counts, edges = np.histogram(ganpimtheta, bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins], axis=1 )
            np.save("lsgan-pimtheta-%d-%d.npy" % (runno, epoch//100), readout)

            counts, edges = np.histogram(ganevents[:, 0], bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins], axis=1 )
            np.save("lsgan-m2pi-%d-%d.npy" % (runno, epoch//100), readout)

            counts, edges = np.histogram(ganevents[:, 1], bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins], axis=1 )
            np.save("lsgan-gamma-%d-%d.npy" % (runno, epoch//100), readout)

            countsx, edgesx = np.histogram(ganevents[:, 2], bins=200)
            binsx = 0.5 * ( edgesx[1:] + edgesx[:-1] )
            countsy, edgesy = np.histogram(ganevents[:, 5], bins=200)
            binsy = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [countsx, binsx, countsy, binsy], axis=1 )
            np.save("lsgan-pxy-%d-%d.npy" % (runno, epoch//100), readout)

            countsx, edgesx = np.histogram(ganevents[:, 3], bins=200)
            binsx = 0.5 * ( edgesx[1:] + edgesx[:-1] )
            countsy, edgesy = np.histogram(ganevents[:, 6], bins=200)
            binsy = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [countsx, binsx, countsy, binsy], axis=1 )
            np.save("lsgan-pipxy-%d-%d.npy" % (runno, epoch//100), readout)

            countsx, edgesx = np.histogram(ganevents[:, 4], bins=200)
            binsx = 0.5 * ( edgesx[1:] + edgesx[:-1] )
            countsy, edgesy = np.histogram(ganevents[:, 7], bins=200)
            binsy = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [countsx, binsx, countsy, binsy], axis=1 )
            np.save("lsgan-pimxy-%d-%d.npy" % (runno, epoch//100), readout)

            counts, edges = np.histogram(ganevents[:, 8], bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins], axis=1 )
            np.save("lsgan-pz-%d-%d.npy" % (runno, epoch//100), readout)

            counts, edges = np.histogram(ganevents[:, 9], bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins], axis=1 )
            np.save("lsgan-pipz-%d-%d.npy" % (runno, epoch//100), readout)
            
            protone = np.sqrt(pmass*pmass + gamma*gamma)
            ganMMP = (ganevents[:, 1] + protone - ganevents[:, 11])*(ganevents[:, 1] + protone - ganevents[:, 11]) - ganevents[:, 2]*ganevents[:, 2] - ganevents[:, 5]*ganevents[:, 5] - ganevents[:, 8]*ganevents[:, 8]
            index = np.where(ganevents[:, 0]>0.08)
            ganMMP_cut = ganMMP[index]

            counts, edges = np.histogram(ganMMP, bins=200)
            bins = 0.5 * ( edges[1:] + edges[:-1] )
            counts_cut, edges_cut = np.histogram(ganMMP_cut, bins=200)
            bins_cut = 0.5 * ( edges[1:] + edges[:-1] )
            readout = np.stack( [counts, bins, counts_cut, bins_cut], axis=1 )
            np.save("lsgan-mmp-%d-%d.npy" % (runno, epoch//100), readout)

        generator.save_weights(imagedir+"generatorfilter_"+str(runno)+"_"+str(epoch//100).zfill(5)+".h5")
        discriminator.save_weights(imagedir+"discriminatorfilter_"+str(runno)+"_"+str(epoch//100).zfill(5)+".h5")
        
        np.save("clasfilter_%d_%d.npy" % (runno, epoch//100), ganevents)
