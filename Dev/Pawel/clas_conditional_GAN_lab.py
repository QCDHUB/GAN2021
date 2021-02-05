#!/usr/bin/env python
import sys,os

import numpy as np
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, ActivityRegularization, Lambda, Concatenate, Permute, Convolution1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.merge import _Merge, concatenate, dot
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from keras import regularizers
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors as mcol
from matplotlib.colors import LogNorm

import pylab as py
#from tools import load, save, checkdir
import pandas as pd
from matplotlib.lines import Line2D


os.environ["CUDA_VISIBLE_DEVICES"]="0"
imagedir = './'


def dot(A,B):
    return A[0]*B[0] - A[1]*B[1] - A[2]*B[2] - A[3]*B[3]


pmass = 0.93827
pipmass = 0.1395
pimmass = 0.1395

epsilon = 0.01

datapath = "data/"
events = X = np.empty(shape=[0, 14])
for i in range(1):
    temp = np.load(datapath+'evts-%d'%(i)+".npy")
    events = np.concatenate([events, temp])

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
cbrtpx = np.cbrt(px)
cbrtpy = np.cbrt(py)
cbrtpz = np.cbrt(pz)
cbrtpipx = np.cbrt(pipx)
# cbrtpipy = np.cbrt(pipy)
cbrtpipz = np.cbrt(pipz)
p = np.stack([px, py, pz], axis=1)
pip = np.stack([pipx, pipy, pipz], axis=1)
pim = np.stack([pimx, pimy, pimz], axis=1)

pxy = px*py
pxz = px*pz
pyz = py*pz
pipxy = pipx*pipy
pipxz = pipx*pipz
pipyz = pipy*pipz
pimxy = pimx*pimy
pimxz = pimx*pimz
pimyz = pimy*pimz

pxpipx = px*pipx
pxpipy = px*pipy
pxpipz = px*pipz
pypipx = py*pipx
pypipy = py*pipy
pypipz = py*pipz
pzpipx = pz*pipx
pzpipy = pz*pipy
pzpipz = pz*pipz

pxpipx = px*pipx
pxpipy = px*pipy
pxpipz = px*pipz
pypipx = py*pipx
pypipy = py*pipy
pypipz = py*pipz
pzpipx = pz*pipx
pzpipy = pz*pipy
pzpipz = pz*pipz

pxpimx = px*pimx
pxpimy = px*pimy
pxpimz = px*pimz
pypimx = py*pimx
pypimy = py*pimy
pypimz = py*pimz
pzpimx = pz*pimx
pzpimy = pz*pimy
pzpimz = pz*pimz

pipxpimx = pipx*pimx
pipxpimy = pipx*pimy
pipxpimz = pipx*pimz
pipypimx = pipy*pimx
pipypimy = pipy*pimy
pipypimz = pipy*pimz
pipzpimx = pipz*pimx
pipzpimy = pipz*pimy
pipzpimz = pipz*pimz

ppip = p*pip
ppim = p*pim
pippim = pip*pim
ppipdot = pe*pipe - px*pipx - py*pipy - pz*pipz
ppimdot = pe*pime - px*pimx - py*pimy - pz*pimz
pippimdot = pipe*pime - pipx*pimx - pipy*pimy - pipz*pimz
lnpe = np.log(pe)
lnpipe = np.log(pipe)

clas_origin = np.stack([gamma, px, py, pz, pipx, pipy, pipz, pimx, pimy, pimz, pe, pipe, pime, M2pi], axis = 1)
# clas_origin = np.concatenate([clas_origin, ppip, ppim, pippim], axis = 1)

# clas_origin = np.stack([px, py, pz, pe], axis = 1)

m2pi2 = (gamma+pmass-pe-pipe)*(gamma+pmass-pe-pipe)\
       -(px+pipx)*(px+pipx)\
       -(py+pipy)*(py+pipy)\
       -(gamma-pz-pipz)*(gamma-pz-pipz)


# normalization
clasmean = np.mean(clas_origin, axis = 0)
classtd = np.std(clas_origin, axis = 0)
clas = (clas_origin - clasmean)/classtd
gfeaturesmean = clasmean[0:7]
gfeaturesstd = classtd[0:7]
emean = clasmean[7]
estd = classtd[7]
gfeaturesmean = clasmean[0:7]
gfeaturesstd = classtd[0:7]
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
M2pimean = clasmean[13:14]
M2pistd = classtd[13:14]
crossmean = clasmean[14:]
crossstd = classtd[14:]

def gen_distributions(evts):

    evts=np.transpose(evts)

    P={}
    P[1]=evts[2]
    P[2]=evts[5]
    P[3]=evts[8]
    P[0]=evts[11]

    pip={}
    pip[1]=evts[3]
    pip[2]=evts[6]
    pip[3]=evts[9]
    pip[0]=evts[12]

    pim={}
    pim[1]=evts[4]
    pim[2]=evts[7]
    pim[3]=evts[10]
    pim[0]=evts[13]


    data={}
    data['P.pip']   = dot(P,pip)
    data['P.pim']   = dot(P,pim)
    data['pim.pim'] = dot(pim,pim)
    data['pip.pim'] = dot(pim,pip)
    data['M2pi']   = evts[0]

    tab=pd.DataFrame(data)


    nrows,ncols=2,2
    fig = plt.figure(figsize=(ncols*4,nrows*3))

    t=tab.query('abs(M2pi-0.019)<0.06')

    ax=plt.subplot(nrows,ncols,1)
    R=(-0.2,0.2)
    ax.hist(tab['M2pi'],bins=100,histtype='step',range=R)
    ax.hist(t['M2pi']  ,bins=100,histtype='step',range=R)
    ax.set_xlabel('missing mass',size=20)

    ax=plt.subplot(nrows,ncols,2)
    #R=(1,2.8)
    R=None#(1,1.8)
    ax.hist(np.sqrt(tab['P.pip']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['P.pip']),bins=100,histtype='step',range=R)
    ax.set_xlabel('P.pi+',size=20)

    ax=plt.subplot(nrows,ncols,3)
    R=None#(1,1.8)
    ax.hist(np.sqrt(tab['P.pim']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['P.pim']),bins=100,histtype='step',range=R)
    ax.set_xlabel('P.pi-',size=20)

    ax=plt.subplot(nrows,ncols,4)
    R=(0,1.8)
    ax.hist(np.sqrt(tab['pip.pim']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['pip.pim']),bins=100,histtype='step',range=R)
    ax.set_xlabel('pi-.pi+',size=20)


    plt.tight_layout()
    checkdir('gallery')
    plt.savefig('gallery/dists.pdf')
    plt.show()

    return

def compare_distributions(evts1, evts2, epoch):

    evts=np.transpose(evts1)

    P={}
    P[1]=evts[2]
    P[2]=evts[5]
    P[3]=evts[8]
    P[0]=evts[11]

    pip={}
    pip[1]=evts[3]
    pip[2]=evts[6]
    pip[3]=evts[9]
    pip[0]=evts[12]

    pim={}
    pim[1]=evts[4]
    pim[2]=evts[7]
    pim[3]=evts[10]
    pim[0]=evts[13]


    data={}
    data['P.pip']   = dot(P,pip)
    data['P.pim']   = dot(P,pim)
    data['pim.pim'] = dot(pim,pim)
    data['pip.pim'] = dot(pim,pip)
    data['M2pi']   = evts[0]

    tab=pd.DataFrame(data)


    nrows,ncols=2,2
    fig = plt.figure(figsize=(ncols*4,nrows*3))

    t=tab.query('abs(M2pi-0.019)<0.06')

    ax=plt.subplot(nrows,ncols,1)
    R=(-0.2,0.2)
    ax.hist(tab['M2pi'],bins=100,histtype='step',range=R)
    ax.hist(t['M2pi']  ,bins=100,histtype='step',range=R)
    ax.set_xlabel('missing mass',size=20)

    ax=plt.subplot(nrows,ncols,2)
    #R=(1,2.8)
    R=None#(1,1.8)
    ax.hist(np.sqrt(tab['P.pip']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['P.pip']),bins=100,histtype='step',range=R)
    ax.set_xlabel('P.pi+',size=20)

    ax=plt.subplot(nrows,ncols,3)
    R=None#(1,1.8)
    ax.hist(np.sqrt(tab['P.pim']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['P.pim']),bins=100,histtype='step',range=R)
    ax.set_xlabel('P.pi-',size=20)

    ax=plt.subplot(nrows,ncols,4)
    R=(0,1.8)
    ax.hist(np.sqrt(tab['pip.pim']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['pip.pim']),bins=100,histtype='step',range=R)
    ax.set_xlabel('pi-.pi+',size=20)

    evts=np.transpose(evts2)

    P={}
    P[1]=evts[2]
    P[2]=evts[5]
    P[3]=evts[8]
    P[0]=evts[11]

    pip={}
    pip[1]=evts[3]
    pip[2]=evts[6]
    pip[3]=evts[9]
    pip[0]=evts[12]

    pim={}
    pim[1]=evts[4]
    pim[2]=evts[7]
    pim[3]=evts[10]
    pim[0]=evts[13]


    data={}
    data['P.pip']   = dot(P,pip)
    data['P.pim']   = dot(P,pim)
    data['pim.pim'] = dot(pim,pim)
    data['pip.pim'] = dot(pim,pip)
    data['M2pi']   = evts[0]

    tab=pd.DataFrame(data)


    nrows,ncols=2,2
    #fig = plt.figure(figsize=(ncols*4,nrows*3))

    t=tab.query('abs(M2pi-0.019)<0.06')

    ax=plt.subplot(nrows,ncols,1)
    R=(-0.2,0.2)
    ax.hist(tab['M2pi'],bins=100,histtype='step',range=R)
    ax.hist(t['M2pi']  ,bins=100,histtype='step',range=R)
    ax.set_xlabel('missing mass',size=20)

    ax=plt.subplot(nrows,ncols,2)
    #R=(1,2.8)
    R=None#(1,1.8)
    ax.hist(np.sqrt(tab['P.pip']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['P.pip']),bins=100,histtype='step',range=R)
    ax.set_xlabel('P.pi+',size=20)

    ax=plt.subplot(nrows,ncols,3)
    R=None#(1,1.8)
    ax.hist(np.sqrt(tab['P.pim']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['P.pim']),bins=100,histtype='step',range=R)
    ax.set_xlabel('P.pi-',size=20)

    ax=plt.subplot(nrows,ncols,4)
    R=(0,1.8)
    ax.hist(np.sqrt(tab['pip.pim']),bins=100,histtype='step',range=R)
    ax.hist(np.sqrt(t['pip.pim']),bins=100,histtype='step',range=R)
    ax.set_xlabel('pi-.pi+',size=20)

    plt.tight_layout()
    checkdir('gallery')
    plt.savefig('gallery/dists'+str(epoch//100).zfill(5)+'.pdf')
    plt.show()

    return

#gen_distributions(events)

HALF_BATCH = 8000
BATCH_SIZE = HALF_BATCH * 2
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper

def MMD_loss(x, y):
    sigma = 0.1
    x1 = x[:HALF_BATCH, :]
    x2 = x[HALF_BATCH:, :]
    y1 = y[:HALF_BATCH, :]
    y2 = y[HALF_BATCH:, :]
    #x1_x2 = K.sum(K.exp(-(x1-x2)*(x1-x2)/sigma))/HALF_BATCH
    #y1_y2 = K.sum(K.exp(-(y1-y2)*(y1-y2)/sigma))/HALF_BATCH
    #x_y = K.sum(K.exp(-(x-y)*(x-y)/sigma))/BATCH_SIZE
    x1_x2 = K.sum(K.exp(sigma/((x1-x2)*(x1-x2)+sigma)))/HALF_BATCH
    y1_y2 = K.sum(K.exp(sigma/((y1-y2)*(y1-y2)+sigma)))/HALF_BATCH
    x_y = K.sum(K.exp(sigma/((x-y)*(x-y)+sigma)))/BATCH_SIZE
    return (x1_x2 + y1_y2 - 2*x_y)*(x1_x2 + y1_y2 - 2*x_y)

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples,gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def _feature_mul(x):
    featuresmean = K.constant(gfeaturesmean)
    featuresstd = K.constant(gfeaturesstd)

    xyz   = x*featuresstd+featuresmean
    gamma = xyz[:, 0:1]
    px    = xyz[:, 1:2]
    py    = xyz[:, 2:3]
    pz    = xyz[:, 3:4]
    pipx  = xyz[:, 4:5]
    pipy  = xyz[:, 5:6]
    pipz  = xyz[:, 6:7]

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

    pe = K.sqrt(pxsq+pysq+pzsq+pmass*pmass)
    pipe = K.sqrt(pipxsq+pipysq+pipzsq+pipmass*pipmass)
    pime = K.sqrt(pimxsq+pimysq+pimzsq+pimmass*pimmass)

    M2pi = (gamma+pmass-pe-pipe)*(gamma+pmass-pe-pipe) - (pimx)*(pimx) - (pimy)*(pimy) - (pimz)*(pimz)

    pim3 = K.concatenate([pimx, pimy, pimz])
    pim3 = (pim3 - K.constant(pimmean))/K.constant(pimstd)

    e = K.concatenate([pe, pipe, pime])
    e = (e - K.constant(emean))/K.constant(estd)

    M2pi = (M2pi - K.constant(M2pimean))/K.constant(M2pistd)

    return K.concatenate([pim3, e, M2pi])

def _make_generator():
    visible = Input(shape=(100,))
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
    output = Dense(7)(LR)
    features = Lambda(feature_mul)(output)

    outputmerge = concatenate([output, features])

    generator = Model(inputs=visible, outputs=[outputmerge])
    return(generator)

def feature_mul(x):
    featuresmean = K.constant(gfeaturesmean)
    featuresstd = K.constant(gfeaturesstd)

    xyz   = x*featuresstd+featuresmean
    gamma = xyz[:, 0:1]
    px    = xyz[:, 0:1]
    py    = xyz[:, 1:2]
    pz    = xyz[:, 2:3]

    P = xyz[:,0:3]
    pipx  = xyz[:, 4:5]
    pipy  = xyz[:, 5:6]
    pipz  = xyz[:, 6:7]

    pimx = -px - pipx
    pimy = -py - pipy
    pimz = gamma - pz - pipz
    #
    pxsq = px*px
    pysq = py*py
    pzsq = pz*pz
    #
    pipxsq = pipx*pipx
    pipysq = pipy*pipy
    pipzsq = pipz*pipz

    pimxsq = pimx*pimx
    pimysq = pimy*pimy
    pimzsq = pimz*pimz
    #
    pe = K.sqrt(pxsq+pysq+pzsq+pmass*pmass)
    pipe = K.sqrt(pipxsq+pipysq+pipzsq+pipmass*pipmass)
    pime = K.sqrt(pimxsq+pimysq+pimzsq+pimmass*pimmass)

    M2pi = (gamma+pmass-pe-pipe)*(gamma+pmass-pe-pipe) - (pimx)*(pimx) - (pimy)*(pimy) - (pimz)*(pimz)

    pim3 = K.concatenate([pimx, pimy, pimz])
    pim3 = (pim3 - K.constant(pimmean))/K.constant(pimstd)

    e = K.concatenate([pe, pipe, pime])
    pe = (pe - K.constant(emean))/K.constant(estd)

    M2pi = (M2pi - K.constant(M2pimean))/K.constant(M2pistd)

    return K.concatenate([pim3, e, M2pi])
    # return pe #K.concatenate([P])

def make_generator():
    visible = Input(shape=(100,))
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
    output = Dense(3)(LR)
    features = Lambda(feature_mul)(output)

    outputmerge = concatenate([output, features])

    generator = Model(inputs=visible, outputs=[outputmerge])
    return(generator)

def make_discriminator():
    visible = Input(shape=(clas.shape[1],))
    hidden1 = Dense(512)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    DR = Dropout(rate=0.1)(LR)
    hidden2 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    DR = Dropout(rate=0.1)(LR)
    hidden3 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden3)
    DR = Dropout(rate=0.1)(LR)
    hidden4 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden4)
    DR = Dropout(rate=0.1)(LR)
    hidden5 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    DR = Dropout(rate=0.1)(LR)
    output = Dense(1)(DR)

    discriminator = Model(inputs=[visible], outputs=output)
    return discriminator

def make_MMD():
    visible = Input(shape=(clas.shape[1],))
    MMD = Model(inputs=visible, output=visible)
    return MMD

def plot_intermediate_stage():

    print(epoch)
    SAMPLE_SIZE = events.shape[0]
    noise= np.random.normal(0, 1, [SAMPLE_SIZE, 100])
    results = generator.predict(noise)
    results = results*classtd+clasmean
    M2pi = results[:, 13]
    gamma = results[:, 0]
    px = results[:, 1]
    py = results[:, 2]
    pz = results[:, 3]
    #px = cbrtpx*cbrtpx*cbrtpx
    #py = cbrtpy*cbrtpy*cbrtpy
    #pz = cbrtpz*cbrtpz*cbrtpz
    pe = results[:, 10]
    pipx = results[:, 4]
    #cbrtpipy = results[:, 6]
    pipy = results[:, 5]
    pipz = results[:, 6]
    #pipx = cbrtpipx*cbrtpipx*cbrtpipx
    #pipy = cbrtpipy*cbrtpipy*cbrtpipy
    #pipz = cbrtpipz*cbrtpipz*cbrtpipz
    pipe = results[:, 11]
    pimx = results[:, 7]
    pimy = results[:, 8]
    pimz = results[:, 9]
    pime = results[:, 12]
    #gamma = np.zeros(SAMPLE_SIZE)
    #pe = np.exp(lnpe)
    #pipe = np.exp(lnpipe)
    ganevents = np.stack([M2pi, gamma, px, pipx, pimx, py, pipy, pimy, pz, pipz, pimz, pe, pipe, pime], axis=1)
    compare_distributions(events, ganevents, epoch)

    plt.hist(events[:, 0], bins=100, range=[-0.3, 0.3], histtype='step')
    plt.hist(ganevents[:, 0], bins=100, range=[-0.3, 0.3], histtype='step')
    plt.title("M2pi")
    plt.show()

    plt.hist(events[:, 1], bins=100, range=[2, 4], histtype='step')
    plt.hist(ganevents[:, 1], bins=100, range=[2, 4], histtype='step')
    plt.title("gamma")
    plt.show()

    plt.hist(events[:, 2], bins=100, range=[-2, 2], histtype='step')
    plt.hist(ganevents[:, 2], bins=100, range=[-2, 2], histtype='step')
    plt.title("px")
    plt.show()

    plt.hist(events[:, 3], bins=100, range=[-2, 2], histtype='step')
    plt.hist(ganevents[:, 3], bins=100, range=[-2, 2], histtype='step')
    plt.title("pipx")
    plt.show()

    plt.hist(events[:, 4], bins=100, range=[-2, 2], histtype='step')
    plt.hist(ganevents[:, 4], bins=100, range=[-2, 2], histtype='step')
    plt.title("pimx")
    plt.show()

    plt.hist(events[:, 5], bins=100, range=[-2, 2], histtype='step')
    plt.hist(ganevents[:, 5], bins=100, range=[-2, 2], histtype='step')
    plt.title("py")
    plt.show()

    plt.hist(events[:, 6], bins=100, range=[-2, 2], histtype='step')
    plt.hist(ganevents[:, 6], bins=100, range=[-2, 2], histtype='step')
    plt.title("pipy")
    plt.show()

    plt.hist(events[:, 7], bins=100, range=[-2, 2], histtype='step')
    plt.hist(ganevents[:, 7], bins=100, range=[-2, 2], histtype='step')
    plt.title("pimy")
    plt.show()

    plt.hist(events[:, 8], bins=100, range=[0, 4], histtype='step')
    plt.hist(ganevents[:, 8], bins=100, range=[0, 4], histtype='step')
    plt.title("pz")
    plt.show()

    plt.hist(events[:, 9], bins=100, range=[-1, 4], histtype='step')
    plt.hist(ganevents[:, 9], bins=100, range=[-1, 4], histtype='step')
    plt.title("pipz")
    plt.show()

    plt.hist(events[:, 10], bins=100, range=[-1, 4], histtype='step')
    plt.hist(ganevents[:, 10], bins=100, range=[-1, 4], histtype='step')
    plt.title("pimz")
    plt.show()

    plt.hist(events[:, 11], bins=100, range=[0, 3.5], histtype='step')
    plt.hist(ganevents[:, 11], bins=100, range=[0, 3.5], histtype='step')
    plt.title("pe")
    plt.show()

    plt.hist(events[:, 12], bins=100, range=[0, 4], histtype='step')
    plt.hist(ganevents[:, 12], bins=100, range=[0, 4], histtype='step')
    plt.title("pipe")
    plt.show()

    plt.hist(events[:, 13], bins=100, range=[0, 4], histtype='step')
    plt.hist(ganevents[:, 13], bins=100, range=[0, 4], histtype='step')
    plt.title("pime")
    plt.show()
    np.save("clasMMD%d.npy"%(epoch//100), ganevents)

def main00():

    generator = make_generator()
    discriminator = make_discriminator()
    MMD = make_MMD()

    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False

    generator_input = Input(shape=(100,))
    generator_layers = generator(generator_input)
    discriminator_layers_for_generator = discriminator(generator_layers)
    MMD_Layers_for_generator = MMD(generator_layers)
    # generator_model = Model(inputs=generator_input,
    #                         outputs=[discriminator_layers_for_generator])
    generator_model = Model(inputs=generator_input,
                            outputs=[discriminator_layers_for_generator,
                                    MMD_Layers_for_generator])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  MMD_loss])
    # generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
    #                         loss=[wasserstein_loss])

    generator_model.summary()
    #sys.exit()

    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples = Input(shape=clas.shape[1:])
    generator_input_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = generator(generator_input_for_discriminator)
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples,
    # to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()([real_samples,
                                                generated_samples_for_discriminator])

    # We then run these samples through the discriminator as well. Note that we never
    # really use the discriminator output for these samples - we're only running them to
    # get the gradient norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input averaged samples to get
    # gradients. However, Keras loss functions can only have two arguments, y_true and
    # y_pred. We get around this by making a partial() of the function with the averaged
    # samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'


    # If we don't concatenate the real and generated samples, however, we get three
    # outputs: One of the generated samples, one of the real samples, and one of the
    # averaged samples, all of size BATCH_SIZE. This works neatly!
    discriminator_model = Model(inputs=[real_samples,
                                        generator_input_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
    # the real and generated samples, and the gradient penalty loss for the averaged samples
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    discriminator_model.summary()


    # We make three label vectors for training. positive_y is the label vector for real
    # samples, with value 1. negative_y is the label vector for generated samples, with
    # value -1. The dummy_y vector is passed to the gradient_penalty loss function and
    # is not used.
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)


    #generator.load_weights("generatorMMD00030.h5")
    #discriminator.load_weights("discriminatorMMD00030.h5")


    for epoch in range(500000):
        np.random.shuffle(clas)
        # print("Number of batches: ", int(dataset.shape[0] // BATCH_SIZE))
        discriminator_loss = []
        generator_loss = []
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        for i in range(int(clas.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
            discriminator_minibatches = clas[i * minibatches_size:
                                                (i + 1) * minibatches_size]

            noise= np.random.normal(0, 1, [BATCH_SIZE*TRAINING_RATIO, 100])

            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
                noise_batch = noise[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                discriminator_loss.append(discriminator_model.train_on_batch(
                    [image_batch, noise_batch],
                    [positive_y, negative_y, dummy_y]))

            noise= np.random.normal(0, 1, [BATCH_SIZE, 100])
            generator_loss.append(generator_model.train_on_batch(noise,
                                                                 [positive_y, image_batch]))
        print(epoch, generator_loss)

        if epoch%1000==0:

        #json_file = 'generatorMMD'+str(epoch//100).zfill(5)+'.json'
        #generator_json = generator.to_json()
        #with open(json_file, "w") as jf:
        #    jf.write(generator_json)

            generator.save_weights(imagedir+"generatorMMD"+str(epoch//100).zfill(5)+".h5")
            discriminator.save_weights(imagedir+"discriminatorMMD"+str(epoch//100).zfill(5)+".h5")


def hist_2d_with_projections( x, y, ax, ax_histx, ax_histy ):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the 2D hist plot:
    bins=300
    ax.hist2d( x, y, bins=bins, cmap='viridis', norm=LogNorm() )

    # XY projections
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

def main01():

    #--matplotlib
    import matplotlib
    # matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    # matplotlib.rc('text',usetex=True)
    import pylab  as py


    generator = make_generator()
    generator.load_weights("generatorMMD00440.h5")

    SAMPLE_SIZE = events.shape[0]
    noise= np.random.normal(0, 1, [SAMPLE_SIZE, 100])
    results = generator.predict(noise)
    results = results*classtd+clasmean
    data =np.transpose(results)

    fig = py.figure( figsize=( 9, 8 ) )
    gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.125, right=0.95, bottom=0.1, top=0.96,
                      wspace=0.02, hspace=0.02)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    hist_2d_with_projections( data[0], data[1], ax, ax_histx, ax_histy )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.set_xlabel( r'$p_x \; [GeV]$', fontsize=18 )
    ax.set_ylabel( r'$p_y \; [GeV]$', fontsize=18 )
    ax_histx.set_title( 'x Projection', fontsize=20 )
    ax_histx.yaxis.tick_right()
    ax_histy.xaxis.tick_top()
    ax_histy.set_ylabel( 'y Projection', fontsize=20, rotation=270. )
    ax_histy.yaxis.set_label_coords(x=1.05, y=0.5, transform=ax_histy.transAxes)

    py.show()
    py.savefig('hist_2d_with_projections.png')

if __name__=="__main__":

    main01()











