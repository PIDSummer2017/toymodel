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

    for _ in xrange(5):
        col = "label" + str(_)
        particletype = df[(col == 1)]
        for entry in xrange(5):
            var = "score" + str(entry)
            plt.hist(particletype[var])
            plt.title("Scores for "+str(entry)+ " when " + str(_) + "is present")
            plt.xlabel('Sigmoid Score')
            plt.yalebl('#')
            plt.savefig('label %d scores %d' % (_, entry))
