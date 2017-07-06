import numpy as np
from dataloader import larcv_data
def time_round(num,digits):
  return float( int(num * np.power(10,digits)) / float(np.power(10,digits)) )

proc = larcv_data()
cfg = {'filler_name': 'DataFiller', 'verbosity':0, 'filler_cfg':'oneclass_filler.cfg'}
proc.configure(cfg)

BATCH_SIZE=20

proc.read_next(BATCH_SIZE)
while 1:
  
  data,label = proc.next()
  proc.read_next(BATCH_SIZE)

  print 'Loaded image data w/ shape:', data.shape
  print 'Loaded label data w/ shape:', label.shape
  print label[0]
  print 'Data  IO time taken:',time_round(proc.time_data_copy / float(proc.read_counter),4),'[s]',
  print '+ reshape',time_round(proc.time_data_conv / float(proc.read_counter),4),'[s]'
  print 'Label IO time taken:',time_round(proc.time_label_copy / float(proc.read_counter),4),'[s]',
  print '+ reshape',time_round(proc.time_label_conv / float(proc.read_counter),4),'[s]'
