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

    electrons = df.loc[df[' label0']==1]
    correctelectrons = electrons.loc[electrons['score00'] >= 0.5]

    z = correctelectrons.entry.count()
    f = electrons.entry.count()

    print str(f) + ' electron count'
    print str(z) + ' electrons correct'

    frac = float(z)/float(f)

    print 'electron accuracy: ' + str(frac)


    protons = df.loc[df[' label4']==1]
    correctprotons = protons.loc[protons['score04'] >= 0.5]

    r = correctprotons.entry.count()
    g = protons.entry.count()

    fracs = float(r)/float(g)

    print 'proton accuracy: ' + str(fracs)

#accuracy('ana-mult-10000.csv')

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


def counter_two_part(f, p1 = 0, p2 = 4, accuracy_comparison = False, two_scores = False, event_plots = False, other_scores = True):

    df = pd.read_csv(f)

    
    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    ep = df.loc[(df[' label'+str(p1)] == 1) & (df[' label'+str(p2)] == 1)]

    excludeds = []
    for i in xrange(5):
        if i != p1 and i != p2:
            part = ' label'+str(i)
            excludeds.append(part)
    print excludeds

    ep_only = ep.loc[(ep[excludeds[0]] == 0) & (ep[excludeds[1]] ==0) & (ep[excludeds[2]]==0)]

    print ep_only.entry.count()
    print df.entry.count()


    good_p1 = ep_only.loc[ep_only['score0'+str(p1)] >= 0.5]
    good_p2 = ep_only.loc[ep_only['score0'+str(p2)] >= 0.5]

    if two_scores:
        plt.figure()
        plt.hist(ep_only['score0'+str(p1)], bins = 50, range = (0., 1.), label = particles[p1] + ' score', alpha = 0.5, normed = True)
        plt.hist(ep_only['score0'+str(p2)], bins = 50, range = (0., 1.), label = particles[p2] + ' score', alpha = 0.5, normed = True)

        plt.legend()
        plt.yscale('log')
        plt.ylabel('event fraction')
        plt.xlabel('scores for events with only '+particles[p1] + ' ' + particles[p2])
        plt.savefig(particles[p1] + particles[p2] + 'multiclassevents.png')
    
    if accuracy_comparison:

        plt.figure()
        plt.hist(good_p1['score0'+str(p2)], bins = 50, range = (0., 1.), label = particles[p2]+ ' score', alpha = 0.4, normed = True)
        plt.xlabel('scores for ' + particles[p2] + ' where ' + particles[p1] + ' is correctly identified')
        plt.yscale('log')
        plt.ylabel('event fraction')
        plt.legend()
        plt.savefig('accurate '+particles[p1] + 'scores for'+particles[p2] + '.png')

        plt.figure()
        plt.hist(good_p2['score0'+str(p1)], bins = 50, range = (0., 1.), label = particles[p1]+ ' score', alpha = 0.4, normed = True)
        plt.xlabel('scores for ' + particles[p1] + ' where ' + particles[p2] + ' is correctly identified')
        plt.yscale('log')
        plt.ylabel('event fraction')
        plt.legend()
        plt.savefig('accurate '+particles[p2] + 'scores for'+particles[p1] + '.png')
    
    if event_plots:

        plt.figure()
        plt.plot(ep_only['score0'+str(p1)],linestyle = 'none', marker = 'o', label = particles[p1])
        plt.plot(ep_only['score0'+str(p2)], label = particles[p2], linestyle = 'none', marker = 'o')
        plt.legend()
        plt.xlabel('scores for events with only '+particles[p1]+' '+particles[p2])
        plt.savefig('entryinfo'+particles[p1]+particles[p2]+'.png')

    if other_scores:
        plt.figure()
        for element in excludeds:
            plt.hist(ep_only['score0'+element[6]], bins = 50, range = (0., 1.), label = particles[float(element[6])], alpha = 0.5)
        plt.xlabel('scores for other particles when only ' +particles[p1] + ' and ' + particles[p2])
        plt.yscale('log')
        plt.ylabel('event fraction')
        plt.legend()
        plt.savefig('otherscores'+particles[p1]+particles[p2]+'.png')OBOB
