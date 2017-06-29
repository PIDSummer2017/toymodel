import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def slice(f):
    """This function looks at the errors which occur in your network training. Input - .csv analysis file from network trianing. Output - descriptions of all network training data, only error data, and a histogram of the most common mistake types made. """

    df = pd.read_csv(f)

    print "Network Labels: " + str(df.describe())

    error_df = df[(df['label'] != df['prediction'])]

    print "Network Errors: " + str(error_df.describe())
    plt.hist2d(df['label'], df['prediction'], norm = LogNorm())
    plt.colorbar()

    plt.title('Network Labels for Batch of 300 after 500 Training Steps')
    plt.xlabel('Image Label')
    plt.ylabel('Network Prediction')
    plt.savefig('labelsvpredictions.png')

slice('analysis.csv')
