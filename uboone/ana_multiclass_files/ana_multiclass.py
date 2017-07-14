import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def accuracy(f):
    df = pd.read_csv(f)

    print df.describe()

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    electrons = df.loc[df['label0']==1]
    correctelectrons = electrons.loc[electrons['score00'] >= 0.5]

    z = correctelectrons.entry.count()
    f = electrons.entry.count()

    frac = float(z)/float(f)

    print 'electron accuracy: ' + str(frac)


    protons = df.loc[df[' label4']==1]
    correctprotons = protons.loc[protons['score04'] >= 0.5]

    r = correctprotons.entry.count()
    g = protons.entry.count()

    fracs = float(r)/float(g)

    print 'proton accuracy: ' + str(fracs)


def exclude(f):
    
    df = pd.read_csv(f)

    no_electrons = df.loc[df['label0']==0]

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    for _ in range(1,5):
        var = 'score0'+str(_)
        label = ' label'+str(_)
        total = no_electrons.loc[no_electrons[label]==1]
        accurate = total.loc[total[var] >= 0.5]
        frac_particle = float(accurate.entry.count())/float(total.entry.count())
        print particles[_] + ' correct fraction w/o electrons '+str(frac_particle) 

def excludeprotons(f):

    df = pd.read_csv(f)

    no_protons = df.loc[df[' label4']==0]

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    for _ in range(4):
        var = 'score0'+str(_)
        label = ' label'+str(_)
        total = no_protons.loc[no_protons[label]==1]
        accurate = total.loc[total[var] >= 0.5]
        frac_particle = float(accurate.entry.count())/float(total.entry.count())
        print particles[_] + ' correct fraction w/o protons '+str(frac_particle)

#excludeprotons('ana-mult-10000.csv')


def counter_ep(f):

    df = pd.read_csv(f)

    
    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    ep = df.loc[(df[' label2'] == 1) & (df[' label4'] == 1)]

    ep_only = df.loc[(df[' label1'] == 0) & (df[' label0'] ==0) & (df[' label3']==0)]

    print ep_only.entry.count()
    print df.entry.count()

counter_ep('ana-mult-10000.csv')
