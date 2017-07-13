#IMPORT NECESSARY PACKAGES
import os,sys
from toy_config import config
#
# Define constants
#
cfg = config()
if not cfg.parse(sys.argv):
  print '[ERROR] Configuraion failure!'
  print 'Exiting...'
  sys.exit(1)

if not cfg.LOAD_FILE:
  print '[ERROR] Analysis requires LOAD_FILE...'
  sys.exit(1)

# Check if log directory already exists
if os.path.isdir(cfg.LOGDIR):
  print '[WARNING] Log directory already present:',cfg.LOGDIR
  user_input=None
  while user_input is None:
    sys.stdout.write('Remove and proceed? [y/n]:')
    sys.stdout.flush()
    user_input = sys.stdin.readline().rstrip('\n')
    if not user_input.lower() in ['y','n','yes','no']:
      print 'Unsupported answer:',user_input
      user_input=None
      continue
  if user_input in ['n','no']:
    print 'Exiting...'
    sys.exit(1)
  else:
    os.system('rm -rf %s' % cfg.LOGDIR)

# Check if chosen network is available
try:
  cmd = 'from toynet import toy_%s' % cfg.ARCHITECTURE
  exec(cmd)
except Exception:
  print 'Architecture',cfg.ARCHITECTURE,'is not available...'
  sys.exit(1)

# Print configuration
print cfg

# ready to import heavy packages
from toynet import toy_lenet
import numpy as np
import tensorflow as tf
from toydata import make_classification_images as make_images

#START ACTIVE SESSION                                                         
sess = tf.InteractiveSession()

#PLACEHOLDERS                                                                 
data_tensor    = tf.placeholder(tf.float32,  [None, 784],name='x')
data_tensor_2d = tf.reshape(data_tensor,[-1,28,28,1])
label_tensor   = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')

#RESHAPE IMAGE IF NEED BE                                                     
tf.summary.image('input',data_tensor_2d,10)

#BUILD NETWORK
net = None
cmd = 'net=toy_%s.build(data_tensor_2d,cfg.NUM_CLASS)' % cfg.ARCHITECTURE
exec(cmd)

#SOFTMAX
with tf.name_scope('softmax'):
  softmax = tf.nn.softmax(logits=net)

#ACCURACY                                                                     
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

#MERGE SUMMARIES FOR TENSORBOARD                                              
merged_summary=tf.summary.merge_all()

#WRITE SUMMARIES TO LOG DIRECTORY LOGS6                                       
writer=tf.summary.FileWriter(cfg.LOGDIR)
writer.add_graph(sess.graph)

sess.run(tf.global_variables_initializer())

saver= tf.train.Saver()
saver = tf.train.import_meta_graph('%s.meta' % cfg.LOAD_FILE)
saver.restore(sess,tf.train.latest_checkpoint('./'))

#
# Run analysis using the trained network
#

# prepare analysis output csv
fout = open('%s/analysis.csv' % cfg.LOGDIR,'w')
fout.write('entry,label,prediction')
for idx in xrange(cfg.NUM_CLASS):
  fout.write(',score%02d' % idx)
fout.write('\n')

# run analysis
from matplotlib import pyplot as plt
batch    = make_images(cfg.BATCH_SIZE,debug=cfg.DEBUG)
score_vv = softmax.eval(feed_dict={data_tensor: batch[0]})
for entry,score_v in enumerate(score_vv):
  label = int(np.argmax(batch[1][entry]))
  prediction = int(np.argmax(score_v))
  fout.write('%d,%d,%d' % (entry, label, prediction))
  for score in score_v:
    fout.write(',%g' % score)
  fout.write('\n')
 
  if cfg.DEBUG and not label == prediction:
    fig, ax = plt.subplots(figsize = (28,28), facecolor = 'w')
    plt.imshow(np.reshape(batch[0][idx], (28, 28)), interpolation = 'none')
    plt.savefig('entry%0d-%d.png' % (idx, label))
    plt.close()

fout.close()

