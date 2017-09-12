import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt

df = pd.read_csv('pid-16599.test1e1pLE_filler_20000.csv')

gammas = df.loc[df['score01']>=0.95]

plt.figure()

plt.hist(gammas['open_angle'], bins = 20, normed = True, label = 'gamma > 0.95, count = '+str(gammas.entry.count()), alpha = 0.4)

plt.hist(df['open_angle'], bins = 20, normed = True, label ='all', alpha = 0.4)
plt.ylim(0., 0.015)
plt.xlabel('open angle')
plt.ylabel('num events')
plt.legend(loc = 4)
plt.savefig('openangleadventure.png')
