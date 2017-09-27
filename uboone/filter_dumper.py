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
import matplotlib.pyplot as plt


########################################
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

##################################
#Need this stuff in order to restore tf session

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

# main part starts here #                                                                                                                                                                                                                                                                                                   
  
# Step 0: configure IO                                                                                                                                                                                                                                                                                                     
# Instantiate and configure                                                                                                                                                                                                                                                                                               
if not cfg.FILLER_CONFIG:                                                                                                                                                                                                                                                                                                   
  print 'Must provide larcv data filler configuration file!'                                                                                                                                                                                                                                                                 
  sys.exit(1)                                                                                                                                                                                                                                                                                                               
proc = larcv_data()                                                                                                                                                                                                                                                                                                          
filler_cfg = {'filler_name': 'DataFiller',                                                                                                                                                                                                                                                                                  
                'verbosity':0,                                                                                                                                                                                                                                                                                               
                'filler_cfg':cfg.FILLER_CONFIG}                                                                                                                                                                                                                                                                              
proc.configure(filler_cfg)                                                                                                                                                                                                                                                                                                   # Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)                                                                                                                                                                                                                 
proc.read_next(cfg.BATCH_SIZE)                                                                                                                                                                                                                                                                                               
# Force data to be read (calling next will sleep enough for the IO thread to finidh reading)                                                                                                                                                                                                                              
proc.next()                                                                                                                                                                                                                                                                                                                 
# Retrieve image/label dimensions                                                                                                                                                                                                                                                                                         
image_dim = proc.image_dim()                                                                                                                                                                                                                                                                                              
label_dim = proc.label_dim()                                                                                                                                                                                                                                                                                                 

data_tensor    = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')                                                                                                                                                                                                                                  
label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')                                                                                                                                                                                                                                           
data_tensor_2d = tf.reshape(data_tensor, [-1,image_dim[2],image_dim[3],1])   
keep_prob = tf.placeholder("float")

# Start IO thread for the next batch while we train the network
proc.read_next(cfg.BATCH_SIZE)


##########################################
#Functions to get out the filters

def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x_1:np.reshape(stimuli,[1,262144]),keep_prob:1.0})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    my_dpi=72
    #filters = np.expand_dims(filters,axis =0)
    plt.figure(1, figsize=(745/my_dpi, 513/my_dpi), dpi=my_dpi)
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        #fig,ax=plt.subplots(figsize=(12,8),facecolor='w')
        #plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        #Problem is arising here from image dimension and between filters shape[3]
        img=plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="jet")
        img.write_png('Filters_Entry'+str(entry)+'_'+str(i)+'.png')
        plt.close()

########################
#Actually Restores Session

sess = tf.InteractiveSession()
# Initialize variables
sess.run(tf.global_variables_initializer())
# Override variables from specified session
saver= tf.train.import_meta_graph('/data/drinkingkazu/summer2017/toymodel/train_multiclass2/pid-18199.meta')
saver.restore(sess,'/data/drinkingkazu/summer2017/toymodel/train_multiclass2/pid-18199')
graph = tf.get_default_graph()
x_1 = graph.get_tensor_by_name("x_1:0") 
 
 
#Get out the names of everything now to find the name of the filter you want
#print(sess.graph.get_operations())
 
# How to access saved operation
op_to_restore11 = graph.get_tensor_by_name("conv1_1/Relu:0")
#op_to_restore12 = graph.get_tensor_by_name("conv1_2/conv1_2_weights/RMSProp_1:0")
#op_to_restore21 = graph.get_tensor_by_name("conv2_1/conv2_1_weights/RMSProp_1:0")
#op_to_restore22 = graph.get_tensor_by_name("conv2_2/conv2_2_weights/RMSProp_1:0")
#op_to_restore31 = graph.get_tensor_by_name("conv3_1/conv3_1_weights/RMSProp_1:0")
#op_to_restore32 = graph.get_tensor_by_name("conv3_2/conv3_2_weights/RMSProp_1:0")
#op_to_restore41 = graph.get_tensor_by_name("conv4_1/conv4_1_weights/RMSProp_1:0")
#op_to_restore42 = graph.get_tensor_by_name("conv4_2/conv4_2_weights/RMSProp_1:0")
#op_to_restore51 = graph.get_tensor_by_name("conv5_1/conv5_1_weights/RMSProp_1:0")
#op_to_restore52 = graph.get_tensor_by_name("conv5_2/conv5_2_weights/RMSProp_1:0")
#op_to_restore = graph.get_tensor_by_name("fc_final/fc_final_weights/RMSProp_1:0")

#Use below print step for debugging to make sure you are restoring a tensor
#print(op_to_restore11)

#########################
#The magic filter step!
imageToUse= adcimg
getActivations(op_to_restore11,adcimg)
#getActivations(op_to_restore12,adcimg)
#getActivations(op_to_restore21,adcimg)
#getActivations(op_to_restore22,adcimg)
#getActivations(op_to_restore31,adcimg)
#getActivations(op_to_restore32,adcimg)
#getActivations(op_to_restore41,adcimg)
#getActivations(op_to_restore42,adcimg)
#getActivations(op_to_restore51,adcimg)
#getActivations(op_to_restore52,adcimg)
#getActivations(op_to_restore,adcimg)










