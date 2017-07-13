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
        col = "label" + str(_)
        particletype = df.loc[df[col]==1]
        #print particletype
        for entry in xrange(5):
            var = "score0" + str(entry)
            print 'column'
            print particletype[var]
            plt.hist(particletype[var])
            plt.title("Scores for "+str(entry)+ " when " + str(_) + "is present")
            plt.xlabel('Sigmoid Score')
            plt.ylabel('#')
            plt.savefig('label %d scores %d' % (_, entry))

slice('analysis.csv')
