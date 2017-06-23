import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('analysis.csv')

#print df.index.size,'entries in csv...'

#print df.describe()

error_df = df[(df['label'] != df['prediction'])]

print 'common errors'
print error_df

print error_df.describe()

labels = [0,1,2,3,4,5,6,7]

vals = []
for i in labels:
    q = len(error_df[(df['label']==i)])
    vals.append(q)

print(vals)

#fig, ax = plt.subplots()
#graph = ax.bar(labels, vals, width = 0.35, color = 'r')
#plt.savefig('errs.png')

from matplotlib.colors import LogNorm

plt.hist2d(df['label'], df['prediction'], norm = LogNorm())
plt.colorbar()

#plt.imshow(H, interpolation = 'nearest')
plt.title('Network Labels for Batch of 300 after 500 Training Steps')
#plt.figure()
#plt.scatter(df['label'], df['prediction'], s=500)
plt.xlabel('Image Label')
plt.ylabel('Network Prediction')
plt.savefig('labelsvpredictions.png')
