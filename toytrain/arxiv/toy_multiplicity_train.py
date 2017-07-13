#IMPORT NECESSAROAY PACKAGESOA
import os,sys
from toy_config import toy_config
#
# Define constants
#
cfg = toy_config()
if not cfg.parse(sys.argv):
  print '[ERROR] Configuraion failure!'
  print 'Exiting...'
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
from toydata import generate_training_images as make_images
from toydata.toydata_varconfig import test_image
#START ACTIVE SESSION                                             \

img = test_image()

sess = tf.InteractiveSession()

#PLACEHOLDERS                                                     \

x = tf.placeholder(tf.float32, [None, 784],name='x')
yvals = []

for shape in range(img.NUM_SHAPES):
  yvals.append(tf.placeholder(tf.float32, [None, 5],name='labels'+str(shape)))

#RESHAPE IMAGE IF NEED BE                                         \

x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input',x_image,5)

#BUILD NETWORK
net = None
cmd = 'net=toy_%s.build(x_image,cfg.NUM_CLASS)' % cfg.ARCHITECTURE
exec(cmd)


#SOFTMAX
with tf.name_scope('softmax'):
  softmax = tf.nn.softmax(logits=net)

#CROSS-ENTROPY                                                    
cross_entropy_total = []
totalerr = None
for idx, label in enumerate(yvals):
  with tf.name_scope('cross_entropy%d' % idx):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=net[idx]))
    cross_entropy_total.append(cross_entropy)
    if totalerr is None:
      totalerr = cross_entropy
    else:
      totalerr += cross_entropy
    tf.summary.scalar('cross_entropy', totalerr)

#CROSS-ENTROPY
#yvals = [y_0, y_1, y_2, y_3]
#cross_entropy_total = []
#totalerr = None
#for idx, label in enumerate(yvals):
 # with tf.name_scope('cross_entropy%d' % idx):
   # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=net[idx])
   # cross_entropy_total.append(cross_entropy)
   # c = np.sum(cross_entropy)
   # print cross_entropy
   # if totalerr is None:
    #  totalerr = c
   # else:
   #   totalerr += c
   # print cross_entropy_total
   # print totalerr
   # tf.summary.scalar('cross_entropy', totalerr)


tominimize = tf.reduce_mean(totalerr)
#TRAINING (RMS OR ADAM-OPTIMIZER OPTIONAL)                        \

with tf.name_scope('train'):
  train_step = tf.train.RMSPropOptimizer(0.0003).minimize(tominimize)

#ACCURACY                        

correct_prediction = []                                 

with tf.name_scope('totalaccuracy'):

    for q in range(img.NUM_SHAPES):
        correct_prediction.append(tf.equal(tf.argmax(net[0],1), tf.argmax(yvals[q],1)))

totalaccuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('total accuracy',totalaccuracy)

with tf.name_scope('square_accuracy'):

  square_accuracy = tf.reduce_mean(tf.cast(correct_prediction[0], tf.float32))

  tf.summary.scalar('square_accuracy', square_accuracy)

with tf.name_scope('tri_accuracy'):

  tri_accuracy = tf.reduce_mean(tf.cast(correct_prediction[1], tf.float32))

  tf.summary.scalar('tri_accuracy', tri_accuracy)

with tf.name_scope('horizontal_accuracy'):

  horizontal_accuracy = tf.reduce_mean(tf.cast(correct_prediction[2], tf.float32))

  tf.summary.scalar('horizontal_accuracy', horizontal_accuracy)

with tf.name_scope('vertical_accuracy'):
  
  vertical_accuracy = tf.reduce_mean(tf.cast(correct_prediction[3], tf.float32))

  tf.summary.scalar('vertical_accuracy', vertical_accuracy)

saver= tf.train.Saver()

sess.run(tf.global_variables_initializer())

#MERGE SUMMARIES FOR TENSORBOARD                                  \

merged_summary=tf.summary.merge_all()

#WRITE SUMMARIES TO LOG DIRECTORY LOGS6                           \

writer=tf.summary.FileWriter(cfg.LOGDIR)
writer.add_graph(sess.graph)

#TRAINING                                                         \

for i in range(cfg.TRAIN_ITERATIONS):

    batch = make_images(cfg.TRAIN_BATCH_SIZE,debug=cfg.DEBUG)

    if i%100 == 0:

        s = sess.run(merged_summary, feed_dict={x:batch[0], yvals[0]:batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]})
        writer.add_summary(s,i)

        train_accuracy = totalaccuracy.eval(feed_dict={x:batch[0], yvals[0]:batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]})
        train_shape_accuracy0 = square_accuracy.eval(feed_dict={x:batch[0], yvals[0]:batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]})
        train_shape_accuracy1 = tri_accuracy.eval(feed_dict={x:batch[0], yvals[0]:batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]})
        train_shape_accuracy2 = vertical_accuracy.eval(feed_dict={x:batch[0], yvals[0]:batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]})
        train_shape_accuracy3 = horizontal_accuracy.eval(feed_dict={x:batch[0], yvals[0]:batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]})
        print("step %d, training accuracy %g, %g, %g, %g, %g"%(i, train_accuracy, train_shape_accuracy0, train_shape_accuracy1, train_shape_accuracy2, train_shape_accuracy3))

    sess.run(train_step,feed_dict={x: batch[0], yvals[0]: batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]})    


    if i%1000 ==0:
        batchtest = make_images(cfg.TEST_BATCH_SIZE,debug=cfg.DEBUG)
        test_accuracy = totalaccuracy.eval(feed_dict={x:batchtest[0], yvals[0]:batchtest[1][0], yvals[1]:batchtest[1][1], yvals[2]:batchtest[1][2], yvals[3]:batchtest[1][3]})
        test_shape_accuracy0 = square_accuracy.eval(feed_dict={x:batchtest[0], yvals[0]:batchtest[1][0], yvals[1]:batchtest[1][1], yvals[2]:batchtest[1][2], yvals[3]:batchtest[1][3]})
        test_shape_accuracy1 = tri_accuracy.eval(feed_dict={x:batchtest[0], yvals[0]:batchtest[1][0], yvals[1]:batchtest[1][1], yvals[2]:batchtest[1][2], yvals[3]:batchtest[1][3]})
        test_shape_accuracy2 = vertical_accuracy.eval(feed_dict={x:batchtest[0], yvals[0]:batchtest[1][0], yvals[1]:batchtest[1][1], yvals[2]:batchtest[1][2], yvals[3]:batchtest[1][3]})
        test_shape_accuracy3 = horizontal_accuracy.eval(feed_dict={x:batchtest[0], yvals[0]:batchtest[1][0], yvals[1]:batchtest[1][1], yvals[2]:batchtest[1][2], yvals[3]:batchtest[1][3]})
        print("step %d, training accuracy %g, %g, %g, %g, %g"%(i, test_accuracy, test_shape_accuracy0, test_shape_accuracy1, test_shape_accuracy2, test_shape_accuracy3))

# post training test
batch = make_images(cfg.TEST_BATCH_SIZE,debug=cfg.DEBUG)
print("Final test accuracy %g"%totalaccuracy.eval(feed_dict={x: batch[0], yvals[0]: batch[1][0], yvals[1]:batch[1][1], yvals[2]:batch[1][2], yvals[3]:batch[1][3]}))

# inform log directory
print('Run `tensorboard --logdir=%s` in terminal to see the result\
s.' % cfg.LOGDIR)

# do analysis, if specified
if not cfg.ANA_BATCH_SIZE:
  sys.exit(0)

#fout = open('%s/analysis.csv' % cfg.LOGDIR,'w')
fout.write('entry,label,prediction')
for idx in xrange(cfg.NUM_CLASS):
  fout.write(',score%02d' % idx)
fout.write('\n')

# run analysis
from matplotlib import pyplot as plt
batch    = make_images(cfg.ANA_BATCH_SIZE,debug=cfg.DEBUG)
score_vv = softmax.eval(feed_dict={x: batch[0]})
for entry,score_v in enumerate(score_vv):
  label = int(np.argmax(batch[1][entry]))
  prediction = int(np.argmax(score_v))
  fout.write('%d,%d,%d' % (entry, label, prediction))
  for score in score_v:
    fout.write(',%g' % score)
  fout.write('\n')

  if not label == prediction:
    fig, ax = plt.subplots(figsize = (28,28), facecolor = 'w')
    plt.imshow(np.reshape(batch[0][idx], (28, 28)), interpolation \
= 'none')
    plt.savefig('entry%0d-%d.png' % (idx, label))
    plt.close()

fout.close()
