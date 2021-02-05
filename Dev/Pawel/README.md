This directory contains GAN(s) developed to tackle CLAS data for two pion photoproduction off a proton.
Each of these codes reads in the data pertaining to the measurement in terms of lab defined 4-vectors.
To reproduce faithfully the experimental data different set of training features are chosen: the lab 4-vectors,
CMS 4-vectors, or a set of invariants that fully describe the reaction final state. Below are some of those variants.

**Lab GAN:**
- clas_conditional_GAN_lab.py - 5 layer original attempt

**Invariant GANs:**
- clas_conditional_GAN_invariant3.ipynb - 5 layer, original
- clas_conditional_GAN_invariant4.ipynb - 5 layer GAN (discriminatorfilter00101.h5, generatorfilter00101.h5)
- clas_conditional_GAN_invariant5.ipynb - 8 layer GAN

**CMS GAN:**
- clas_conditional_GAN_cm2_orig.ipynb - 8 layer GAN
- clas_conditional_GAN_cm2.ipynb - 20 layer GAN 

Scripts/notebooks run well with:
**tensorflow 2.1.0**,
**Keras 2.3.0** 
<br>Most recent versions of TensorFlow/Keras break gradient penatlty part in loss functions.

A data file, evts-clas.npy, these notebooks need to run is located in /work/JAM/pawel
