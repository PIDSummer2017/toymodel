import sys
from toytrain import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from time import gmtime, strftime



cfg = config()
if not cfg.parse(sys.argv) or not cfg.sanity_check():
    sys.exit(1)

df_test  = pd.read_csv('test_csv/plane%s/test_plane%s.csv'%(cfg.PLANE,cfg.PLANE))
df_train = pd.read_csv('test_csv/plane%s/train_plane%s.csv'%(cfg.PLANE,cfg.PLANE))

t=strftime("%Y-%m-%d %H:%M:%S", gmtime()) 

fig, ax = plt.subplots(1,1,figsize=(10,8))
ax.plot(df_train.iter.values, df_train.acc.values, '-*',color='blue', label='Train_sample Acc')
ax.plot(df_test.iter.values, df_test.acc.values, '-*',color='red' ,label='Test_sample Acc')
ax1=ax.twinx()
ax1.plot(df_train.iter.values, df_train.loss.values, '-*',color='orange', label='Train_sample Loss')

ax.set_ylabel('Accuracy')
ax.set_xlabel('Step')
ax1.set_ylabel('Loss')

ax.grid()

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1))
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=20)
ax1.legend(bbox_to_anchor=(1.05, 0.1), loc=2, borderaxespad=0., fontsize=20)
ax.set_title("Traning Status %s"%t)

fig.savefig('Rui_monitor.png',bbox_inches="tight")

