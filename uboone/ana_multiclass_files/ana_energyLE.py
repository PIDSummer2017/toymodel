import numpy as np
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import pandas as pd

df  = pd.read_csv('pid-16599.test1e1pLE_filler_20000.csv')

plt.figure()
H, bedges = np.histogram(df['max_energy00'], bins = 15, range = (25., 105.))
plt.bar(bedges[:-1], H, width = 80./15., yerr = np.sqrt(H), color = 'b', alpha = 0.6)
plt.xlabel('electron energy (MeV)')
plt.ylabel('number of events')
plt.savefig('sampleenegry.png') 
