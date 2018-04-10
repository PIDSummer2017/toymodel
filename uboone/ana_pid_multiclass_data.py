# Basic imports
import os,sys,time
from toytrain import config
import numpy as np
import tensorflow as tf

#
# Utility functions
#
import numpy as np
# Integer rounder
def time_round(num,digits):
  return float( int(num * np.power(10,digits)) / float(np.power(10,digits)) )

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

def main():

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
  from dataloader import larcv_data

  #########################
  # main part starts here #
  #########################

  #
  # Step 0: configure IO
  #

  # Instantiate and configure
  if not cfg.FILLER_CONFIG:
    print 'Must provide larcv data filler configuration file!'
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
  # Retrieve image/label dimensions
  image_dim = proc.image_dim()
  label_dim = proc.label_dim()

  #
  # Step 1: prepare truth information handle
  #
  from larcv import larcv
  from ROOT import TChain
  filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
  roi_chain = TChain("partroi_p0roi_pid_tree")
  for fname in filler.pd().io().file_list():
    roi_chain.AddFile(fname)
  #filler.set_next_index(5861)
  # Immediately start the thread for later IO
  proc.read_next(cfg.BATCH_SIZE)

  #
  # Step 2: Build network
  #

  # Set input data and label for training
  print 'image_dim is ', image_dim

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
  # Step 3: Configure global process (session, summary, etc.)
  #
  # Create a session

  session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
  #sess = tf.Session(config=session_conf)
  sess = tf.InteractiveSession(config=session_conf)
  # Initialize variables
  sess.run(tf.global_variables_initializer())
  # Override variables if wished
  reader=tf.train.Saver()
  reader.restore(sess,cfg.LOAD_FILE)
  # Analysis csv file
  weight_file_name = cfg.LOAD_FILE.split('/')[-1]
  filler_file_name = cfg.FILLER_CONFIG.split('/')[-1].replace('.cfg','')
  fout = open('%s.%s.plane%s.csv' % (weight_file_name,filler_file_name,cfg.PLANE),'w')
  fout.write('entry,run,subrun,event')
  for idx in xrange(cfg.NUM_CLASS):
    fout.write(',score%02d'%idx)
  fout.write('\n')

  # Run training loop
  entry_number_v = [0] * cfg.BATCH_SIZE
  this_entry = 0
  for i in range(cfg.ITERATIONS):
    # Report the progress
    sys.stdout.write('Processing %d/%d\r' % (i,cfg.ITERATIONS))
    sys.stdout.flush()
    # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
    data,label = proc.next()
    processed_entries = filler.processed_entries()
    for entry in xrange(processed_entries.size()):
      entry_number_v[entry] = processed_entries[entry]
    # Run loss & train step
    score_vv = sess.run(sigmoid,feed_dict={data_tensor: data})
                
    for res_idx,score_v in enumerate(score_vv):
      this_entry+=1
      if(this_entry> roi_chain.GetEntries()): break;
      entry = entry_number_v[res_idx]
      fout.write('%d' % entry)

      info = get_info(roi_chain, entry)
      fout.write(',%d' % (info.run))
      fout.write(',%d' % (info.subrun))
      fout.write(',%d' % (info.event))

      for score in score_v:
        fout.write(',%g' % score)
      fout.write('\n')
    if(this_entry> roi_chain.GetEntries()): break;
    # Start IO thread for the next batch 

    proc.read_next(cfg.BATCH_SIZE)
  fout.close()
  print
  print 'Done'

if __name__ == '__main__':
  from choose_gpu import pick_gpu
  GPUMEM=10000
  GPUID=pick_gpu(GPUMEM,caffe_gpuid=True)
  if GPUID < 0:
    sys.stderr.write('No available GPU with memory %d\n' % GPUMEM)
    sys.exit(1)
  #with tf.device('/gpu:%d' % GPUID):
  with tf.device('/cpu:0'):
    main()
