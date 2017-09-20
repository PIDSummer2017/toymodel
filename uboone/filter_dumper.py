import matplotlib
matplotlib.use('Agg')

import ROOT
from larcv import larcv
larcv.load_pyutil
from ROOT import TChain
import sys
import numpy as np
from toytrain import config
import os,sys,time
import tensorflow as tf
from dataloader import larcv_data
import math

#Below Code gets out the specified Image

ch=TChain("image2d_wire_tree")
ch.AddFile(sys.argv[1])

entry = int(sys.argv[2])
zoom = 0
if len(sys.argv)>3:
    zoom = int(sys.argv[3])

print ch.GetEntries()
import matplotlib.pyplot as plt

ch.GetEntry(entry)

br = ch.image2d_wire_branch

image2d =  br.Image2DArray()[2]

adcimg = larcv.as_ndarray(image2d)

nz_pixels=np.where(adcimg>40.0)
ylim = (np.min(nz_pixels[0])-20,np.max(nz_pixels[0])+20)
xlim = (np.min(nz_pixels[1])-20,np.max(nz_pixels[1])+20)

fig,ax=plt.subplots(figsize=(12,8),facecolor='w')
img=plt.imshow(adcimg,cmap='jet',vmin=0,vmax=300,interpolation=None)
if zoom:
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
img.write_png('Entry%05d.png' % entry)
plt.close()

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

###############
data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')                                                                                                                                                                                                                                  
label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')                                                                                                                                                                                                                                           
data_tensor_2d = tf.reshape(data_tensor, [-1,image_dim[2],image_dim[3],1])   
keep_prob = tf.placeholder("float")

  # Start IO thread for the next batch while we train the network
proc.read_next(cfg.BATCH_SIZE)
##############
def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={data_tensor:np.reshape(stimuli,[1,262144]),keep_prob:1.0})
    units = np.expand_dims(units, axis=3)
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
   # filters = np.expand_dims(filters,axis =0)
    plt.figure(1, figsize=(100,100))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        fig,ax=plt.subplots(figsize=(12,8),facecolor='w')
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        img=plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="jet")
        img.write_png('Filters_Entry%05d.png' % entry)
        plt.close()

##############
#
#def main():

  # Load configuration and check if it's good
#cfg = config()
#if not cfg.parse(sys.argv) or not cfg.sanity_check():
#  sys.exit(1)
  
  # Print configuration
#print '\033[95mConfiguration\033[00m'
#print cfg
#time.sleep(0.5)

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
#if not cfg.FILLER_CONFIG:
  #print 'Must provide larcv data filler configuration file!'
 # sys.exit(1)

#proc = larcv_data()
#filler_cfg = {'filler_name': 'DataFiller', 
                #'verbosity':0, 
                #'filler_cfg':cfg.FILLER_CONFIG}
#proc.configure(filler_cfg)
  # Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
#proc.read_next(cfg.BATCH_SIZE)
  # Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
#proc.next()
  # Retrieve image/label dimensions
#image_dim = proc.image_dim()
#label_dim = proc.label_dim()

  #
  # Step 1: prepare truth information handle
  #
#from larcv import larcv
#from ROOT import TChain
#filler = larcv.ThreadFillerFactory.get_filler("DataFiller")
#roi_chain = TChain("partroi_segment_tree")
#for fname in filler.pd().io().file_list():
#  roi_chain.AddFile(fname)
#filler.set_next_index(5861)
  # Immediately start the thread for later IO
#proc.read_next(cfg.BATCH_SIZE)

  #
  # Step 2: Build network
  #

  # Set input data and label for training
#data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')
#label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')
#data_tensor_2d = tf.reshape(data_tensor, [-1,image_dim[2],image_dim[3],1])

  # Call network build function (then we add more train-specific layers)
#net = None
#cmd = 'from toynet import toy_%s;net=toy_%s.build(data_tensor_2d,cfg.NUM_CLASS)' % (cfg.ARCHITECTURE,cfg.ARCHITECTURE)
#exec(cmd)

sess = tf.InteractiveSession()
  # Initialize variables
sess.run(tf.global_variables_initializer())
  # Override variables if wished
#reader=tf.train.Saver()
#reader.restore(sess,cfg.LOAD_FILE)

saver= tf.train.import_meta_graph('/data/drinkingkazu/summer2017/toymodel/train_multiclass2/pid-18199.meta')
saver.restore(sess,'/data/drinkingkazu/summer2017/toymodel/train_multiclass2/pid-18199')
graph = tf.get_default_graph()

#for op in tf.get_default_graph().get_operations():
#    print str(op.name) 

print(sess.graph.get_operations())
#How to access saved variable/Tensor/placeholders 
#conv1_1 = graph.get_tensor_by_name("conv1_1_weights_1:0")
 
## How to access saved operation
op_to_restore = graph.get_tensor_by_name("fc_final/fc_final_weights/RMSProp_1:0")

#########################
imageToUse= adcimg
getActivations(op_to_restore,adcimg)
#print(getActivations)
#print(sess)







