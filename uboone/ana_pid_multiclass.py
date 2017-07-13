# Basic imports
import os,sys,time
from toytrain import config

# Load configuration and check if it's good
cfg = config()
if not cfg.parse(sys.argv) or not cfg.sanity_check():
  sys.exit(1)

# Print configuration
print '\033[95mConfiguration\033[00m'
print cfg
time.sleep(0.5)

# Import more libraries (after configuration is validated)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
from dataloader import larcv_data

#
# Utility functions
#
# Integer rounder
def time_round(num,digits):
  return float( int(num * np.power(10,digits)) / float(np.power(10,digits)) )

#########################
# main part starts here #
#########################

#
# Step 0: configure IO
#

# Instantiate and configure
proc = larcv_data()
filler_cfg = {'filler_name': 'DataFiller', 
              'verbosity':0, 
              'filler_cfg':'%s/uboone/multiclass_filler.cfg' % os.environ['TOYMODEL_DIR']}
proc.configure(filler_cfg)
# Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
proc.read_next(cfg.BATCH_SIZE)
# Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
proc.next()
# Immediately start the thread for later IO
proc.read_next(cfg.BATCH_SIZE)
# Retrieve image/label dimensions
image_dim = proc.image_dim()
label_dim = proc.label_dim()

#
# 1) Build network
#

# Set input data and label for training
data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')
label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')
data_tensor_2d = tf.reshape(data_tensor, [-1,image_dim[2],image_dim[3],1])

# Call network build function (then we add more train-specific layers)
net = None
cmd = 'from toynet import toy_%s;net=toy_%s.build(data_tensor_2d,cfg.NUM_CLASS)' % (cfg.ARCHITECTURE,cfg.ARCHITECTURE)
exec(cmd)

# Define accuracy
with tf.name_scope('sigmoid'):
  sigmoid = tf.nn.sigmoid(net)

#
# 2) Configure global process (session, summary, etc.)
#
# Create a session
sess = tf.InteractiveSession()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Override variables if wished
reader=tf.train.Saver()
reader.restore(sess,cfg.LOAD_FILE)
# Analysis csv file
fout = open('%s/analysis.csv' % cfg.LOGDIR,'w')
fout.write('entry,label0, label1, label2, label3, label4')
for idx in xrange(cfg.NUM_CLASS):
  fout.write(',score%02d' % idx)
fout.write('\n')  
# Run training loop
for i in range(cfg.ITERATIONS):
  # Report the progress
  sys.stdout.write('Processing %d/%d\r' % (i,cfg.ITERATIONS))
  sys.stdout.flush()
  # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
  data,label = proc.next()
  # Start IO thread for the next batch while we train the network
  proc.read_next(cfg.BATCH_SIZE)
  # Run loss & train step
  score_vv = sess.run(sigmoid,feed_dict={data_tensor: data, label_tensor: label})

  for entry,score_v in enumerate(score_vv):
    fout.write('%d' % (entry + i * cfg.BATCH_SIZE))
    for item in xrange(cfg.NUM_CLASS):
      labelz = label[entry][item]
      fout.write(',%d' % (labelz))
    for score in score_v:
      fout.write(',%g' % score)
    fout.write('\n')
fout.close()
print
print 'Done'
proc.reset()
