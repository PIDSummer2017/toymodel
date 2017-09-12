import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def accuracy(f):
    df = pd.read_csv(f)

    #print df.describe()

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    for i in xrange(5):
        plt.figure()
        #plt.hist(df["score0"+str(i)], bins = 20, range = (0., 1.), alpha = 0.5)
        plt.xlabel(particles[i]+ " score for 1e1p LE events")
        plt.ylabel('number of events')
        plt.yscale('log')
        nbins = 15
        H, bin_edges = np.histogram(df['score0'+str(i)], bins = nbins, range = (0., 1.))
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        
        #plt.errorbar(bin_centers, H, yerr = errs, fmt = 'None', linewidth = 6.0, color = 'r')
        plt.xlim(0., 1.)
        plt.bar(bin_edges[:-1], H, yerr = np.sqrt(H), width = 1.0/nbins, alpha = 0.5, color = 'c')
        plt.savefig("1e1pLE "+particles[i]+" scorenew.png")

    from matplotlib.colors import LogNorm
    plt.figure()
    plt.hist2d(df["score00"], df["score02"], bins = 50, norm = LogNorm())
    plt.colorbar()
    plt.xlabel("electron score for 1e1p events")
    plt.ylabel("muon score for 1e1p events")
    plt.plot(np.arange(0., 1., 0.01), np.arange(0., 1., 0.01), marker = 'None', color = 'r', linewidth = 4., linestyle = '--')
    plt.savefig("1e1p-emuhist-new.png", interpolation = "None")

    plt.figure()
    plt.hist2d(df["score00"], df["score01"], bins = 50, norm = LogNorm())
    plt.colorbar()
    plt.xlabel("electron score for 1e1p events")
    plt.ylabel("gamma score  for 1e1p events")
    plt.plot(np.arange(0., 1., 0.01), np.arange(0., 1., 0.01), marker= 'None', color = 'r', linewidth = 4., linestyle = '--')
    plt.savefig("1e1p-egammahist-new.png", interpolation = "None")

accuracy('pid-18199.test1e1pLE_filler.csv')

def energy_range(f):
    df = pd.read_csv(f)

    particles = {0:'electron', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}

    plt.figure()
    plt.hist2d(df['score01'].astype(float), df['dcosz00'].astype(float), bins = 20, norm = LogNorm())
    plt.colorbar()
    plt.xlabel('photon score for 1e1p events')
    plt.ylabel('dcos for 1e1p events')
    plt.savefig('gammascorevsdcos1e1p00.png', interpolation = 'None')

    plt.figure()
    plt.hist2d(df['score02'], df['max_energy00'], bins = 20, norm = LogNorm())
    plt.colorbar()
    plt.xlabel('muon score for 1e1p events')
    plt.ylabel('max electron energy for 1e1p events')
    plt.savefig('muonscorevsenergy1e1p00.png', interpolation = 'None')

#energy_range('pid-16599.test1e1p00_filler.csv')

def dcosplot(f):
    df = pd.read_csv(f)
    plt.figure()
    plt.hist(df['dcosz00'], bins = 50, range = (-1., 1.), normed = True)
    plt.xlabel('electron dcos z')
    plt.ylabel('event fraction')
    plt.savefig('test1e1pLEdcosz00-00.png')

#dcosplot('pid-16599.test1e1pLE_filler_full.csv')

def energy_for_muon_errors(f):

    df = pd.read_csv(f)

    muons = df.loc[df["score02"] >= 0.7]

    #print muons

    plt.figure()
    H, edges = np.histogram(df['max_energy00'], bins = 15, normed = True, range = (30., 100.))
    plt.bar(edges[:-1], H, yerr = np.sqrt(H)/df.entry.count(), color = 'c', label = 'all LE 1e1p (count = '+str(df.entry.count())+')', alpha = 0.3, width = 70./15.)
    
    plt.xlabel('electron energy')
    plt.xlabel('electron energy')
    plt.ylabel('fraction of events')
    Hmu, edgesmu = np.histogram(muons['max_energy00'], normed = True,  bins = 15, range = (30., 100.))
    plt.bar(edgesmu[:-1], Hmu, width = 70./15., yerr = (np.sqrt(Hmu)/muons.entry.count()), color = 'm', label = 'muon score >0.7 (count = '+str(muons.entry.count())+')', alpha = 0.3)
    plt.legend()
    plt.legend()
    plt.savefig('electronenergyformuonfalsepositivesLE-new.png')
   # print H, Hmu

    plt.figure()
    H, edges = np.histogram(df['max_energy00'], bins = 20, normed = True, range = (25., 100.))
    plt.bar(edges[:-1], H, yerr = np.sqrt(H)/df.entry.count(), color = 'b', label = 'all LE 1e1p (count = '+str(df.entry.count())+')', alpha = 0.3, width = 75./20.)
    print 'range' + str(df.max_energy00.max()) +',' + str(df.max_energy00.min())
    plt.xlabel('electron energy (MeV)')
    plt.ylabel('fraction of events')
    plt.savefig('energy-new.png')
energy_for_muon_errors('pid-18199.test1e1pLE_filler.csv')

def pion_differentiation(f):
    df = pd.read_csv(f)
    plt.figure()
    plt.hist2d(df['score03'], df['max_energy00'], bins = 50, norm = LogNorm())

    plt.xlabel('pion score for 1e1p events')
    plt.ylabel('max electron energy')
    plt.colorbar()
    plt.savefig('pionscorevselectron00.png')
    
    pions = df.loc[df['score03']>=0.5]
    
    plt.figure()
    plt.hist(pions['max_energy00'], bins = 20, normed = True, label = 'pion score > 0.5 (count = '+str(pions.entry.count())+')', alpha = 0.3)
    plt.hist(df['max_energy00'], bins = 20, normed = True, label = 'all events (count = 20000)', alpha = 0.3)
    plt.xlabel('max electron energy')
    plt.ylabel('event fraction')
    plt.legend()
    plt.savefig('pionenergy-electrondistribution00.png')


    plt.figure()
    plt.hist2d(df['score04'], df['max_energy04'], bins = 50, norm=LogNorm())
    plt.xlabel('proton score for 1e1p events')
    plt.ylabel('proton max energy, 1e1p events')
    plt.savefig('protonscorevsenergy00.png')

#pion_differentiation('pid-16599.test1e1p00_filler.csv')

def angle_resolution():
    plt.figure()
    df  = pd.read_csv('pid-18199.test1e1pLE_filler.csv')
    z = df.query('dcosz00 >= -1')
    gammaprime = z.loc[z['score01']<=0.9]
    gammaz = gammaprime.loc[gammaprime['score01']>=0.5]
    z['anglez00']=z.apply(lambda x: np.arccos(x['dcosz00']) / np.pi * 180, axis=1)
    gammaz['anglez00']=gammaz.apply(lambda x: np.arccos(x['dcosz00']) / np.pi * 180, axis=1)
    plt.hist(z.anglez00, bins=10,normed =True, range=(0, 180), alpha = 0.3, label = 'all events angle - count = 20000')
    plt.hist(gammaz.anglez00, bins = 10, normed = True, range = (0, 180), alpha = 0.3, label = 'gamma score > 0.9, count = '+str(gammaz.entry.count())+')')
    plt.xlabel('angle')
    plt.ylabel('event fraction')
    plt.legend(loc = 4)
    plt.savefig('gammavsangleLE-lessbins-09cutoffmoregammasLE-new.png')
 
 #   print gammaz.entry()
    #plt.figure()
    #plt.hist2d(z['anglez00'], z['score01'], bins = 50, norm = LogNorm())
 #   plt.xlabel('electron angle from z')
 #   plt.ylabel('photon score')
 #   plt.colorbar()
 #   plt.savefig('electronanglevsphotonscore2dhist00.png')


   # plt.figure()
    #plt.hist(gammaz['anglez00'],bins = 20, label  = 'gamma score between 0.5 and .9, count ='+str(gammaz.entry.count()), normed = True)
    #plt.hist(z['anglez00'], bins = 20, label = 'all scores', normed = True)
    #plt.xlabel('electron energy')
    #plt.ylabel('frac')
    #plt.legend()
    #plt.savefig('midrangegammaenergy.png')

    plt.figure()
    g, bedges = np.histogram(gammaz['anglez00'],bins = 15, range = (0., 180.), normed = True)
    plt.bar(bedges[:-1], g, width = 180./15.,color = 'b', alpha = 0.4, label = 'gamma score > 0.5, count ='+str(gammaz.entry.count()), yerr= (np.sqrt(g)/gammaz.entry.count()))
    h, medges = np.histogram(z['anglez00'], bins = 15, range = (0., 180.), normed = True)
    plt.bar(medges[:-1], h, color = 'g', alpha = 0.4, label = 'all events, count = 20,000', yerr = (np.sqrt(h)/z.entry.count()), width = 180/15.)
    plt.xlabel('angle from normal to wire direction')
    plt.ylabel('event fraction')
    plt.legend(loc = 4)
    plt.savefig('gammammorethan5-new.png')


    gammas = z.loc[z['score01']>=0.95]
    print gammas.entry
    plt.figure()
    
    gammas['anglez00']=gammas.apply(lambda x: np.arccos(x['dcosz00']) / np.pi * 180, axis=1)
    c, cedges = np.histogram(gammas['anglez00'],bins = 15, range = (0., 180.), normed = True)
    plt.bar(cedges[:-1], c, width = 180./15., alpha = 0.4, color = 'b', label = 'gamma score > 0.95, count ='+str(gammas.entry.count()), yerr = (np.sqrt(c)/gammas.entry.count()))
    p, pedges = np.histogram(z['anglez00'], bins = 15, range = (0., 180.), normed = True)
    plt.bar(pedges[:-1], p, alpha = 0.4, label = 'all events, count = 20,000', color = 'g', yerr = (np.sqrt(p)/z.entry.count()), width = 180./15.)
    plt.xlabel('angle from normal to wire direction')
    plt.ylabel('event fraction')
    plt.legend(loc = 3)
    plt.savefig('gammammorethan95-new.png')

    #interesting = gammas.loc[gammas['anglez00']<=20]
#
  #  print interesting.entry

angle_resolution()

def photon_energy(f):
    df = pd.read_csv(f)

    gammas = df.loc[df['score01'] >= 0.5]

    plt.figure()

    plt.hist(gammas['max_energy00'], bins = 50, range = (0.,1.), label = 'gamma score > 0.5, count = ' + str(gammas.entry.count()), alpha = 0.4)
    plt.hist(df['max_energy00'], bins = 50, range = (0., 1.), label = 'all data, count = ' + str(df.entry.count()),alpha = 0.4)

    
    plt.xlabel('electron energy for 1e1p 00 events')
    plt.ylabel('event fraction')
    plt.legend()

    plt.savefig('photonvselectronenergy00.png')

#photon_energy('pid-16599.test1e1p00_filler.csv')
B
