import sys,os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from time import gmtime, strftime

def make_plot(ax, plane):
    
    name=sys.argv[2]

    df_test  = pd.read_csv('/data/dayajun/toymodel/uboone/test_csv/plane%s/%s/test_plane%s.csv'%(plane,name,plane))
    df_train = pd.read_csv('/data/dayajun/toymodel/uboone/test_csv/plane%s/%s/train_plane%s.csv'%(plane,name,plane))
    
    t=strftime("%Y-%m-%d %H:%M:%S", gmtime())

    ax1=ax.twinx()

    ln1 = ax1.plot(df_train.iter.values, df_train.loss.values, '-*',color='orange', label='Train_sample Loss', zorder = 0 )

    #ax1.set_zorder(0)

    ln2 = ax.plot(df_train.iter.values,  df_train.acc.values,  '-*',color='blue',   label='Train_sample Acc', zorder = 1)
    ln3 = ax.plot(df_test.iter.values,   df_test.acc.values,   '-*',color='red' ,   label='Test_sample Acc', lw=5, zorder = 1)
    #ax.set_zorder(1)

    ax1.set_ylabel('Loss')
    ax1.set_ylim(0,2)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Step')


    ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    #lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1))
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    #ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    #ax1.legend(bbox_to_anchor=(0., 1.09, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    #ax.set_title("%s Traning Status %s"%(name,t))


def main():
    
    name=sys.argv[2]
    '''
    fig, ax = plt.subplots(1,1,figsize=(8,6))

    plane=0

    F_test_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/%s/test_plane%s.csv'%(plane,name,plane))
    F_train_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/%s/train_plane%s.csv'%(plane,name,plane))
    
    if (not F_test_file*F_train_file):
        ax.axis([0,10,0,10])
        
        if (not F_test_file): ax.text(2,3,'No test csv',fontsize=30)
        if (not F_train_file): ax.text(2,7,'No training csv',fontsize=30)
        
    make_plot(ax, plane)

    '''
    #fig, (ax_1,ax_2,ax_3) = plt.subplots(3,1,figsize=(10,24))
    fig_1, ax_1 = plt.subplots(1,1,figsize=(8,6))
    fig_2, ax_2 = plt.subplots(1,1,figsize=(8,6))
    fig_3, ax_3 = plt.subplots(1,1,figsize=(8,6))
    figs=[fig_1,fig_2,fig_3]

    axes=[ax_1,ax_2,ax_3]

    plane=0
    for ax in axes:
        #F_test_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/test_plane%s.csv'%(plane,plane))
        #F_train_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/train_plane%s.csv'%(plane,plane))
        F_test_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/%s/test_plane%s.csv'%(plane,name,plane))
        F_train_file = os.path.isfile('/data/dayajun/toymodel/uboone/test_csv/plane%s/%s/train_plane%s.csv'%(plane,name,plane))
        
        if (not F_test_file*F_train_file):
            ax.axis([0,10,0,10])

            if (not F_test_file): ax.text(2,3,'No test csv',fontsize=30)
            if (not F_train_file): ax.text(2,7,'No training csv',fontsize=30)

            continue

        make_plot(ax, plane)


        figs[plane].savefig('Monitor_%s_%i.png'%(name,plane),bbox_inches="tight")
        plane+=1
        
    
if __name__ == '__main__':
    main()
