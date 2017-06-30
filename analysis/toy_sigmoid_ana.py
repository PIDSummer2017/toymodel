import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def read_in(f):

    df = pd.read_csv(f)
    return df
    print "Network Labels: " + str(df.describe())

def errorslice(f):
    """This function looks at the errors which occur in your network trainin\
g. Input - .csv analysis file from network training. Output - descriptions o\
f all network training data, only error data, and a histogram of the most co\
mmon mistake types made. """

    df = read_in(f)

    error_df = df[(df['label'] != df['prediction'])]

    print "Network Errors: " + str(error_df.describe())
    plt.hist2d(df['label'], df['prediction'], norm = LogNorm())
    plt.colorbar()

    plt.title('Network Labels for Batch of 300 after 500 Training Steps')
    plt.xlabel('Image Label')
    plt.ylabel('Network Prediction')
    plt.savefig('labelsvpredictions.png')

def two_shape_imgs(f):
    df = pd.read_csv(f)
    print "Network Labels: " + str(df.describe())
    label_array = [0,0,0,0]
    for shape in range(3):
        for othershape in range(3):
            if shape != othershape:
                twoshape_df = df[(df['label' + str(shape)] == 1)&(df['label' + str(othershape)] == 1)]
                print str(shape) + ' ' + str(othershape) + ' summary:' + str(combined_df.describe())
                print str(shape) + ' ' + str(othershape) + ' errors:' + str(error_twoshape.describe())
                plt.hist(twoshape_df['score'+str(shape)], norm = LogNorm())
                plt.colorbar
                plt.title('Sigmoid Training Labels for ' + str(shape) + str(othershape) + ' images')
                plt.xlabel('Image Label')
                plt.ylabel('Network Prediction')
                plt.savefig('errs_%d.png') % (str(shape) + str(othershape))

two_shape_imgs('sigmoid_ana.csv')
