import matplotlib
matplotlib.use('Agg')

import ROOT
from larcv import larcv
larcv.load_pyutil
from ROOT import TChain
import sys
import numpy as np

ch=TChain("image2d_wire_tree")
ch.AddFile(sys.argv[1])

entry = int(sys.argv[2])

zoom = 0
if len(sys.argv)>3:
    zoom = int(sys.argv[3])

print ch.GetEntries()
import matplotlib.pyplot as plt

ch.GetEntry(entry)

br = ch.image2d_wire_branch

image2d =  br.Image2DArray()[2]

adcimg = larcv.as_ndarray(image2d)

nz_pixels=np.where(adcimg>40.0)
ylim = (np.min(nz_pixels[0])-20,np.max(nz_pixels[0])+20)
xlim = (np.min(nz_pixels[1])-20,np.max(nz_pixels[1])+20)

fig,ax=plt.subplots(figsize=(12,8),facecolor='w')
img=plt.imshow(adcimg,cmap='jet',vmin=0,vmax=400,interpolation=None)
ax.set_ylim(*ylim)
ax.set_xlim(*xlim)
img.write_png('Entry%05d.png' % entry)
plt.close()
