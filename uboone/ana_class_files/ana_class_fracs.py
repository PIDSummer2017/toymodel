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

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    accuracies = []

    for _ in xrange(5):
        particletype = df.loc[df['label']==_]
        correctguess = particletype.loc[particletype['label']==particletype['prediction']]
    
        z = correctguess.entry.count()
        f = particletype.entry.count()
        frac =  float(z)/float(f)
        print z, f
        
        print frac
        accuracies.append(frac)
  #      print frac
#        print particles[_] + ' overall'
 #       print particletype.describe()
  #      print particles[_] + 'correct'
   #     print correctguess.describe()

    for i in range(len(accuracies)):
       print particles[i] +' classification accuracy: '+ str(accuracies[i])

slice('ana-10000.csv')
