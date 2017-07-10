#IMPORT NECESSARY PACKAGES
import os,sys,time
from toytrain import toy_config
#
# Define constants
#
cfg = toy_config()
if not cfg.parse(sys.argv):
  print '[ERROR] Configuraion failure!'
  print 'Exiting...'
  sys.exit(1)

# Check if chosen network is available
try:
  cmd = 'from toynet import toy_%s' % cfg.ARCHITECTURE
  exec(cmd)
except Exception:
  print 'Architecture',cfg.ARCHITECTURE,'is not available...'
  sys.exit(1)

# Print configuration
print '\033[95mConfiguration\033[00m'
print cfg
time.sleep(0.5)

# ready to import heavy packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from toynet import toy_lenet
import numpy as np
import tensorflow as tf
import numpy as np
from dataloader import larcv_data
def time_round(num,digits):
  return float( int(num * np.power(10,digits)) / float(np.power(10,digits)) )
proc = larcv_data()
filler_cfg = {'filler_name': 'DataFiller',
              'verbosity':0,
              'filler_cfg':'%s/uboone/multiclass_filler.cfg' % os.environ['TOYMODEL_DIR']}


proc.configure(filler_cfg)
proc.read_next(cfg.TRAIN_BATCH_SIZE)
proc.next()
proc.read_next(cfg.TRAIN_BATCH_SIZE)
image_dim = proc.image_dim()
label_dim = proc.label_dim()

#START ACTIVE SESSION                                                 \

sess = tf.InteractiveSession()

#PLACEHOLDERS                                                         \

x  = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')
y_ = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')

#RESHAPE IMAGE IF NEED BE                                             \

x_image = tf.reshape(x, [-1,image_dim[2],image_dim[3],1])
tf.summary.image('input',x_image,10)

#BUILD NETWORK
net = None
cmd = 'net=toy_%s.build(x_image,cfg.NUM_CLASS)' % cfg.ARCHITECTURE
exec(cmd)

#SOFTMAX
with tf.name_scope('sigmoid'):
  sigmoid = tf.nn.sigmoid(net)

#ACCURACY
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.rint(sigmoid), tf.rint(y_))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

#MERGE SUMMARIES FOR TENSORBOARD
merged_summary=tf.summary.merge_all()

saver= tf.train.Saver()

sess.run(tf.global_variables_initializer())
saver= tf.train.Saver()
saver = tf.train.import_meta_graph('%s.meta' % cfg.ANA_FILE)
saver.restore(sess,tf.train.latest_checkpoint('./'))


fout = open('%s/analysis.csv' % cfg.LOGDIR,'w')
fout.write('entry,label0, label1, label2, label3')
for idx in xrange(cfg.NUM_CLASS):
  fout.write(',score%02d' % idx)
fout.write('\n')

# post training test
#data,label = proc.next()
#proc.read_next(cfg.TEST_BATCH_SIZE)
#data,label = proc.next()
#print("Final test accuracy %g"%accuracy.eval(feed_dict={x: data, y_: label}))

from matplotlib import pyplot as plt

data,label = proc.next()
proc.read_next(cfg.TEST_BATCH_SIZE)
data,label = proc.next()
print np.shape(data[0])
print label

for element in xrange(cfg.TRAIN_BATCH_SIZE):
  score_vv = sigmoid.eval(feed_dict={x:data[element]})
  for entry,score_v in enumerate(score_vv):
    fout.write('%d' % (entry))
    for item in xrange(cfg.NUM_CLASS):
      labelz = label[entry][item]
      fout.write('%d' % (labelz))
    for score in score_v:
      fout.write(',%g' % score)
    fout.write('\n')

    for i in range(4):
      if not np.int(score_v[i]+0.5) == batch[1][entry][i]:
        plt.figure()
        plt.imshow(np.reshape(batch[0][entry], (28,28)), interpolation = 'nearest')
        plt.savefig(str(entry)+str(batch[1][entry])+str(score_v)+'.png')
        plt.close()

fout.close()
