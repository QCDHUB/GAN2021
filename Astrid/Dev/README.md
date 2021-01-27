# Development of code for Ylm generation from CLAS and GAN data.

This folder will include and be updated with all the codes needed for generating and comparing Ylm.

The main notebook is Ylm\_Lab.ipynb, currently set up for the lab-frame GAN. It calls the external Python codes mceg.py and tools.py. Furthermore, it uses generator files gamma\_hist.npy, generatorfilter00224.h5, gfeaturesmean.npy, gfeaturesstd.npy, as provided by Yaohang.

In order to work completely, it will need access also  to local directories with CLAS data, and where to store large npy arrays.
