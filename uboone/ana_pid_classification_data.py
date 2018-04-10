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
# Classification label conversion
def convert_label(input_label,num_class):
  result_label = np.zeros((len(input_label),num_class))
  for idx,label in enumerate(input_label):
    result_label[idx][int(label)]=1.
  return result_label
# define entry info
class entry_info:
  def __init__(self):
    self.run = 0
    self.subrun = 0
    self.event = 0
# info retriever
def get_info(roi_chain, entry):
  roi_chain.GetEntry(entry)
  roi = roi_chain.partroi_p0roi_pid_branch
  res = entry_info()
  res.run = roi.run()
  res.subrun = roi.subrun()
  res.event = roi.event()
  return res
#########################
# main part starts here #
#########################

#
# Step 0: configure IO
#

# Instantiate and configure
if not cfg.FILLER_CONFIG:
  'Must provide larcv data filler configuration file!'
  sys.exit(1)
proc = larcv_data()
filler_cfg = {'filler_name': 'DataFiller', 
              'verbosity':0, 
              'filler_cfg':cfg.FILLER_CONFIG}
proc.configure(filler_cfg)
# Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
proc.read_next(cfg.BATCH_SIZE)
# Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
proc.next()

from larcv import larcv
from ROOT import TChain
filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
roi_chain = TChain("partroi_p0roi_pid_tree")
for fname in filler.pd().io().file_list():
  roi_chain.AddFile(fname)
filler.set_next_index(0)

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

# Define loss
with tf.name_scope('softmax'):
  softmax = tf.nn.softmax(logits=net)

# Define accuracy
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#
# 2) Configure global process (session, summary, etc.)
#
# Create a session
sess = tf.InteractiveSession()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Create a summary writer handle
writer=tf.summary.FileWriter(cfg.LOGDIR)
writer.add_graph(sess.graph)
# Load trained parameters
reader=tf.train.Saver()
reader.restore(sess,cfg.LOAD_FILE)
# Prepare output CSV file
fout = open('%s/analysis.csv' % cfg.LOGDIR,'w')
fout.write('entry,run,subrun,event,label')
for idx in xrange(cfg.NUM_CLASS):
  fout.write(',score%02d' % idx)
fout.write(',prediction')
fout.write('\n')
  
# Run analysis loop
entry_number_v = [0] * cfg.BATCH_SIZE
this_entry = 0
for i in range(cfg.ITERATIONS):
  # Report the progress
  sys.stdout.write('Processing %d/%d\r' % (i,cfg.ITERATIONS))
  sys.stdout.flush()
  # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
  data,label = proc.next()
  # Get entry #
  processed_entries = filler.processed_entries()
  for entry in xrange(processed_entries.size()):
    entry_number_v[entry] = processed_entries[entry]
  # Start IO thread for the next batch while we train the network
  proc.read_next(cfg.BATCH_SIZE)
  # Run loss & train step
  score_vv = sess.run(softmax,feed_dict={data_tensor: data})
  # Record to csv
  for res_idx,score_v in enumerate(score_vv):
    entry = entry_number_v[res_idx]
    info = get_info(roi_chain, entry)

    this_entry+=1
    if (this_entry > roi_chain.GetEntries()): break;

    fout.write('%d' % (res_idx))
    fout.write(',%d' % (info.run))
    fout.write(',%d' % (info.subrun))
    fout.write(',%d' % (info.event))
    fout.write(',%d' % (label[res_idx]))
    for score in score_v:
      fout.write(',%g' % score)
    fout.write(',%g' % np.argmax(score_v))
    fout.write('\n')
  if(this_entry> roi_chain.GetEntries()): break;
fout.close()
print
print 'Done'
proc.reset()
