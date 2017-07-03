#IMPORT NECESSARY PACKAGE
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
  print '[ERROR] Log directory already present:',cfg.LOGDIR
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
print cfg

# ready to import heavy packages
from toynet import toy_lenet
import numpy as np
import tensorflow as tf
from toydata import generate_training_images as make_images

#START ACTIVE SESSION                                                      
sess = tf.InteractiveSession()

#PLACEHOLDERS                                                                 
x = tf.placeholder(tf.float32, [None, 784],name='x')
y_ = tf.placeholder(tf.float32, [None, 4],name='labels')

#print(x)
#print(y_)
#RESHAPE IMAGE IF NEED BE                                                     
x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input',x_image,10)

#BUILD NETWORK
net = None
cmd = 'net=toy_%s.build(x_image,cfg.NUM_CLASS)' % cfg.ARCHITECTURE
exec(cmd)

#SIGMOID
with tf.name_scope('softmax'):

  softmax = tf.nn.sigmoid(net)

#CROSS-ENTROPY                                                                
with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=net))
  tf.summary.scalar('cross_entropy',cross_entropy)

#TRAINING (RMS OR ADAM-OPTIMIZER OPTIONAL)                                    
with tf.name_scope('train'):
  train_step = tf.train.RMSPropOptimizer(0.0003).minimize(cross_entropy)

#ACCURACY                                                                     
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.rint(net), tf.rint(y_))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

saver= tf.train.Saver()

sess.run(tf.global_variables_initializer())

#MERGE SUMMARIES FOR TENSORBOARD                                           
merged_summary=tf.summary.merge_all()

#WRITE SUMMARIES TO LOG DIRECTORY LOGS6                                    
writer=tf.summary.FileWriter(cfg.LOGDIR)
writer.add_graph(sess.graph)

#TRAINING                                                                  
for i in range(cfg.TRAIN_ITERATIONS):


    batch = make_images(num_images = cfg.TRAIN_BATCH_SIZE,debug=cfg.DEBUG, bad_label=False)

    #print(batch[1])
    #print(batch)
    if i%100 == 0:
        
        s = sess.run(merged_summary, feed_dict={x:batch[0], y_:batch[1]})
        writer.add_summary(s,i)
        
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    
        print("step %d, training accuracy %g"%(i, train_accuracy))

    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})                                    

    if i%1000 ==0:
        batchtest = make_images(cfg.TEST_BATCH_SIZE,debug=cfg.DEBUG, bad_label=False)
        test_accuracy = accuracy.eval(feed_dict={x:batchtest[0], y_:batchtest[1]})
        print("step %d, test accuracy %g"%(i, test_accuracy))

# post training tes
batch = make_images(cfg.TEST_BATCH_SIZE,debug=cfg.DEBUG, bad_label=False)
print("Final test accuracy %g"%accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))


prediction=tf.argmax(net,1)
print "Predictions", prediction.eval(feed_dict={x:batch[0]}, session=sess)
print "Correct Predictions", correct_prediction.eval({x:batch[0], y_:batch[1]}, session=sess)

# inform log directory
print('Run `tensorboard --logdir=%s` in terminal to see the results.' % cfg.LOGDIR)

# do analysis, if specified
if not cfg.ANA_BATCH_SIZE:
  sys.exit(0)

#
# Run analysis using the trained network
#

# prepare analysis output csv
fout = open('%s/analysis.csv' % cfg.LOGDIR,'w')
fout.write('entry,label0, label1, label2, label3')
for idx in xrange(cfg.NUM_CLASS):
  fout.write(',score%02d' % idx)
fout.write('\n')

# run ana
batch    = make_images(cfg.ANA_BATCH_SIZE,debug=cfg.DEBUG,bad_label=False)
score_vv = softmax.eval(feed_dict={x: batch[0]})
for entry,score_v in enumerate(score_vv):
  label0 = batch[1][entry][0]
  label1 = batch[1][entry][1]
  label2 = batch[1][entry][2]
  label3 = batch[1][entry][3]
  #prediction0 = score_v[0]
  #prediction1 = score_v[1]
  #prediction2 = score_v[2]
 # prediction3 = score_v[3]
  for score in score_v:
    fout.write(',%g' % score)
  fout.write('%d, %d, %d, %d, %d' % (entry, label0, label1, label2, label3))
  fout.write('\n')

fout.close()
