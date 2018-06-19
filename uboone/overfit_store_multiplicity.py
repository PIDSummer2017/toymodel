import os,sys,glob,time
import numpy as np
import pandas as pd
import root_numpy as rn

from toytrain import config

font = {'size'   : 14}
particle_dic = {0:'eminus', 1:'gamma', 2:'muon', 3:'pion', 4:'proton'}
cfg=config()
if not cfg.parse(sys.argv) or not cfg.sanity_check():
    print 'Need config file'
    sys.exit(1)
list_of_files = glob.glob('test_csv/plane%s/multiplicity/*'%cfg.PLANE)
latest_file = max(list_of_files, key=os.path.getctime)
print latest_file
step = latest_file.split('/')[3].split('-')[1].split('.')[0]

print "latestfile, ", latest_file
print "step",latest_file.split('/')[3].split('-')[1].split('.')[0]


df=pd.read_csv(latest_file)

df = df.drop_duplicates(subset=['entry'])

##Create csv file for one plane
if not (os.path.isfile('test_csv/plane%s/multiplicity/test_plane%s.csv'%(cfg.PLANE,cfg.PLANE))):
    fout = open('test_csv/plane%s/multiplicity/test_plane%s.csv' %(cfg.PLANE,cfg.PLANE),'w')
    fout.write('iter,acc')
    fout.write('\n')
else:
    fout = open('test_csv/plane%s/multiplicity/test_plane%s.csv' %(cfg.PLANE,cfg.PLANE),'a')

#Calculate particle presence accu and multiplicity accu
df_test = df

ptypes=['eminus','gamma','muon','pion','proton']



def get_multi_accuracy(row):
    true_multi=np.zeros(25)
    true_multi[int(row.label0)]    = 1
    true_multi[int(row.label1+5)]  = 1
    true_multi[int(row.label2+10)] = 1
    true_multi[int(row.label3+15)] = 1
    true_multi[int(row.label4+20)] = 1

    predicted_multi=np.zeros(25)
    for x in xrange(25):
        idx = int(x/5.)
        ptype=ptypes[idx]
        mul = x%5
        cmd = 'predicted_multi[x]=row.%s_multi_%i'%(ptype, mul)
        exec(cmd)
    predicted_multi = np.rint(predicted_multi)

    numerator   = np.sum(np.equal(predicted_multi, true_multi)*true_multi)

    denomenator = np.sum(true_multi)
    
    return numerator/denomenator
    
    
df_test['multiplicity_accu'] = df_test.apply(lambda row: get_multi_accuracy(row), axis=1)

multi_accu = np.sum(df_test.multiplicity_accu.values)/(df_test.index.size)




fout.write('\n')
fout.write('%s,%s'%(step,multi_accu))

