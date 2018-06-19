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
list_of_files = glob.glob('test_csv/plane%s/pid/*'%cfg.PLANE)
latest_file = max(list_of_files, key=os.path.getctime)
step = latest_file.split('/')[3].split('-')[1].split('.')[0]

print "latestfile, ", latest_file
print "step",latest_file.split('/')[3].split('-')[1].split('.')[0]


df=pd.read_csv(latest_file)

print df.index.size
df = df.drop_duplicates(subset=['entry'])
print df.index.size

##Create csv file for one plane
if not (os.path.isfile('test_csv/plane%s/pid/test_plane%s.csv'%(cfg.PLANE,cfg.PLANE))):
    fout = open('test_csv/plane%s/pid/test_plane%s.csv' %(cfg.PLANE,cfg.PLANE),'w')
    fout.write('iter,acc')
    fout.write('\n')
else:
    fout = open('test_csv/plane%s/pid/test_plane%s.csv' %(cfg.PLANE,cfg.PLANE),'a')

#Calculate particle presence accu and multiplicity accu
df=pd.read_csv(latest_file)

df = df.drop_duplicates(subset=['entry'])

df_m2 = df#.query("multi_sum==1")

df_m2['pred0'] = 0
df_m2['pred1'] = 0
df_m2['pred2'] = 0
df_m2['pred3'] = 0
df_m2['pred4'] = 0
df_m2 = df_m2.reset_index()

def fill_pred(row, par_idx):
    scores = np.array([row.score00,row.score01,row.score02,row.score03,row.score04])
    scores = np.rint(scores)
    if (scores[par_idx]==1 ):
        return 1
    else:
        return 0

df_m2['pred0']=df_m2.apply(lambda row: fill_pred(row,0), axis=1)
df_m2['pred1']=df_m2.apply(lambda row: fill_pred(row,1), axis=1)
df_m2['pred2']=df_m2.apply(lambda row: fill_pred(row,2), axis=1)
df_m2['pred3']=df_m2.apply(lambda row: fill_pred(row,3), axis=1)
df_m2['pred4']=df_m2.apply(lambda row: fill_pred(row,4), axis=1)

def get_presence(row, label_idx):
    labels = np.array([row.label0, row.label1, row.label2, row.label3, row.label4])
    if (labels[label_idx]>=1):
        return 1
    else:
        return 0

df_m2['presence0']=df_m2.apply(lambda row: get_presence(row,0), axis=1)
df_m2['presence1']=df_m2.apply(lambda row: get_presence(row,1), axis=1)
df_m2['presence2']=df_m2.apply(lambda row: get_presence(row,2), axis=1)
df_m2['presence3']=df_m2.apply(lambda row: get_presence(row,3), axis=1)
df_m2['presence4']=df_m2.apply(lambda row: get_presence(row,4), axis=1)

def reduce_mean(row):
    labels=np.array([row.label0,row.label1,row.label2,row.label3,row.label4])
    preds=np.array([row.pred0, row.pred1, row.pred2, row.pred3, row.pred4])
    equal=np.equal(labels, preds)
    reduce_mean=np.mean(equal)
    return  reduce_mean

df_m2['accuracy']=0
df_m2['accuracy']=df_m2.apply(lambda row: reduce_mean(row), axis=1)

def reduce_mean_presence(row):
    labels=np.array([row.presence0,row.presence1,row.presence2,row.presence3,row.presence4])
    preds=np.array([row.pred0, row.pred1, row.pred2, row.pred3, row.pred4])
    equal=np.equal(labels, preds)
    reduce_mean=np.mean(equal)
    return  reduce_mean

df_m2['accuracy_presence']=0
df_m2['accuracy_presence']=df_m2.apply(lambda row: reduce_mean_presence(row), axis=1)

def presence_multiplicity(row):
    labels=np.array([row.presence0,row.presence1,row.presence2,row.presence3,row.presence4])
    return np.sum(labels)
df_m2['presence_multiplicity']=df_m2.apply(lambda row: presence_multiplicity(row), axis=1)

pre_accu = np.sum(df_m2.accuracy_presence.values)/df_m2.index.size
mul_accu = np.sum(df_m2.accuracy.values)/df_m2.index.size

#print 'Presence Accuracy: ', pre_accu
#print 'Multiplicity Accuracy: ', mul_accu

fout.write('\n')
fout.write('%s,%s'%(step,pre_accu))

