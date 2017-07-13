import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def slice(f):
    df = pd.read_csv(f)

    print df.describe()

    print df

    for _ in xrange(5):
        particletype = df.loc[df['label']==_]
        correctguess = particletype.loc[particletype['label']==particletype['prediction']]
                                        
        frac = float(len(particletype)/len(correctguess))
        print frac

slice('ana-10000.csv')
