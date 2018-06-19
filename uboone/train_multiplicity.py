# Basic imports
import os,sys,time
import shutil,csv
from toytrain import config

# Load configuration and check if it's good
cfg = config()
if not cfg.parse(sys.argv) or not cfg.sanity_check():
  sys.exit(1)

# Get the start iter number
start_iter=0
if cfg.LOAD_FILE:
  start_iter=int(cfg.LOAD_FILE.split('-')[1])

# Print configuration
print '\033[95mConfiguration\033[00m'
print cfg
time.sleep(0.5)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Import more libraries (after configuration is validated)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
from dataloader import larcv_data

# Create a csv file for online monitor
if not (os.path.isfile('test_csv/plane%s/multiplicity/train_plane%s.csv'%(cfg.PLANE,cfg.PLANE))):
  print 'Creating a new csv file for plane %s'%cfg.PLANE
  fout = open('test_csv/plane%s/multiplicity/train_plane%s.csv'%(cfg.PLANE,cfg.PLANE),'w')
  fout.write('iter,acc,loss')
  fout.write('\n')
  fout.flush()
else:
  print 'Found a csv file for plane %s'%cfg.PLANE
  #fout = open('test_csv/plane%s/multiplicity/train_plane%s.csv'%(cfg.PLANE,cfg.PLANE),'r+')
  #lines = fout.read().split("\n")
  fout = open('test_csv/plane%s/multiplicity/train_plane%s.csv'%(cfg.PLANE,cfg.PLANE),'w')
  '''
  word = str(start_iter)
  line_del = 0
  for i,line in enumerate(lines):
    if word in line: # or word in line.split() to search for full words
      print("Word \"{}\" found in line {}".format(word, i+1))
      line_del=i+1
  print 'line number larger than ',line_del, ' are deleted'
  for x in range(len(lines)):
    if x<line_del:
      fout.write(lines[x])
      fout.write('\n')
  fout.truncate()
  '''

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
# Immediately start the thread for later IO
proc.read_next(cfg.BATCH_SIZE)
# Retrieve image/label dimensions
image_dim = proc.image_dim()
label_dim = proc.label_dim()
multiplicity_dim = proc.multiplicity_dim()

#
# 1) Build network
#

# Set input data and label for training
#with tf.device('/gpu:%i'%cfg.GPU_INDEX):
#keep_prob = tf.placeholder(tf.float32)
data_tensor           = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')
label_tensor          = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')
multiplicity_tensor   = tf.placeholder(tf.float32, [None, cfg.MULTIPLICITY_CLASS],name='multiplicities')
data_tensor_2d        = tf.reshape(data_tensor, [-1,image_dim[2],image_dim[3],1])
tf.summary.image('input',data_tensor_2d,10)

# Call network build function (then we add more train-specific layers)
train_net = None
test_net  = None
print 'cfg.ARCHITECTURE is ',cfg.ARCHITECTURE
cmd = 'from toynet import toy_%s;train_net=toy_%s.build(data_tensor_2d,cfg.NUM_CLASS,trainable=True,reuse=False,keep_prob = 0.5)' % (cfg.ARCHITECTURE,cfg.ARCHITECTURE)
exec(cmd)
cmd = 'from toynet import toy_%s;test_net =toy_%s.build(data_tensor_2d,cfg.NUM_CLASS,trainable=False,reuse=True,keep_prob = 1.0)' % (cfg.ARCHITECTURE,cfg.ARCHITECTURE)
exec(cmd)

for tensor in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
  print tensor.name
  tf.summary.histogram(tensor.name, tensor)

# Define accuracy
with tf.name_scope('accuracy'):
  sigmoid = tf.nn.sigmoid(test_net)
  tf.summary.histogram('sigmoid', sigmoid)
    
  predicted_multi = tf.rint(sigmoid)
  true_multi      = tf.rint(multiplicity_tensor)
  correct_prediction      = tf.cast(tf.equal(predicted_multi, true_multi), tf.float32)
  correct_prediction_norm = tf.multiply(correct_prediction, true_multi) 

  numerator   = tf.reduce_sum(correct_prediction_norm)
  denominator = tf.cast(tf.reduce_sum(true_multi), tf.float32)

  accuracy = tf.divide(numerator, denominator)
  tf.summary.scalar('accuracy', accuracy)
  # Define loss + backprop as training step
with tf.name_scope('train'):

  ##precision, pre_op = tf.metrics.precision(labels=label_tensor, predictions=train_net)
  true_multi      = multiplicity_tensor
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=multiplicity_tensor, logits=train_net)
  cross_entropy = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy',cross_entropy)
  ##cross_entropy = tf.divide(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_tensor, logits=train_net)), precision)
  train_step = tf.train.RMSPropOptimizer(0.0001).minimize(cross_entropy)  
  #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
  #
  # 2) Configure global process (session, summary, etc.)
  #
  # Create a bandle of summary
merged_summary=tf.summary.merge_all()
# Create a session
#sess = tf.InteractiveSession()
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
# Initialize variables
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
# Create a summary writer handle
log_path = cfg.LOGDIR+str(cfg.PLANE)
#writer=tf.summary.FileWriter(cfg.LOGDIR)
writer=tf.summary.FileWriter(log_path)
writer.add_graph(sess.graph)
saver=tf.train.Saver(max_to_keep=1000)
# Override variables if wished
if cfg.LOAD_FILE:
  vlist=[]
  avoid_params=cfg.AVOID_LOAD_PARAMS.split(',')
  for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    if v.name in avoid_params:
      print '\033[91mSkipping\033[00m loading variable',v.name,'from input weight...'
      continue
    print '\033[95mLoading\033[00m variable',v.name,'from',cfg.LOAD_FILE
    vlist.append(v)
  reader=tf.train.Saver(var_list=vlist)
  reader.restore(sess,cfg.LOAD_FILE)
  
# Run training loop
for i in range(cfg.ITERATIONS):
  i=i+start_iter
  # Receive data (this will hang if IO thread is still running = this will wait for thread to finish & receive data)
  data,label,multiplicity = proc.next()
  # Start IO thread for the next batch while we train the network
  proc.read_next(cfg.BATCH_SIZE)
  # Run loss & train step

  #tmp0 = tf.Print(test_net,     [test_net], "test_net")
  #tmp1 = tf.Print(sigmoid,      [sigmoid], "sigmoid")
  #tmp2 = tf.Print(label_tensor, [label_tensor], "true label")
  #loss,acc,_,print0,print1,print2 = sess.run([cross_entropy,accuracy,train_step,tmp0,tmp1,tmp2],feed_dict={data_tensor: data, label_tensor: label})
  loss,acc,_= sess.run([cross_entropy,accuracy,train_step],feed_dict={data_tensor: data, multiplicity_tensor: multiplicity})

  sys.stdout.write('Training in progress @ step %d loss %g accuracy %g\r' % (i,loss,acc))
  sys.stdout.flush()

  # Debug mode will dump images
  if cfg.DEBUG:
    shutil.rmtree('multi_debug')
    if not os.path.exists('multi_debug'):
      os.makedirs('multi_debug')
    for idx in xrange(len(data)):
      img = None 
      img = data[idx].reshape([image_dim[2],image_dim[3]])
      
      adcpng = plt.imshow(img)
      imgname = 'multi_debug/debug_class_%d_entry_%04d.png' % (np.argmax(label[idx]),i*cfg.ITERATIONS+idx)
      if os.path.isfile(imgname): raise Exception
      adcpng.write_png(imgname)
      plt.close()

      print '%-3d' % (i*cfg.ITERATIONS+idx),'...',
      print 'shape',img.shape,
      print img.min(),'=>', img.max(),'...',
      print img.mean(),'+/-',img.std(),'...',
      print 'max loc @',np.unravel_index(img.argmax(),img.shape),'...',
      print imgname

  # If configured to save summary + snapshot, do so here.
  if (i+1)%cfg.SAVE_ITERATION == 0:
    # Run summary
    s = sess.run(merged_summary, feed_dict={data_tensor:data, multiplicity_tensor:multiplicity})
    writer.add_summary(s,i)
    # Save snapshot
    #ssf_path = saver.save(sess,cfg.ARCHITECTURE,global_step=i)
  
    print 'iter is ',i
    print 'loss is ',loss
    print 'acc is ',acc
    print fout
    fout.write('%d,%g,%g'%(i,acc,loss))
    fout.write('\n')
    fout.flush()
    save_path = os.path.join("plane%itraining/multiplicity"%cfg.PLANE)
    ssf_path = saver.save(sess,save_path+'/'+cfg.ARCHITECTURE,global_step=i)

    #save_path = os.path.join("plane%itraining"%cfg.PLANE)
    #epoch_number = int(50.*i/24990)
    #ssf_path = saver.save(sess,save_path+'/'+cfg.ARCHITECTURE,global_step=epoch_number)
    print 'saved @',ssf_path

fout.close()

# post training test
data,label,multiplicity = proc.next()
proc.read_next(cfg.BATCH_SIZE)
data,label,multiplicity = proc.next()
print("Final test accuracy %g"%accuracy.eval(feed_dict={data_tensor: data, multiplicity_tensor: multiplicity}))

# inform log directory
print('Run `tensorboard --logdir=%s` in terminal to see the results.' % log_path)