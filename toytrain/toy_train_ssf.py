#IMPORT NECESSARY PACKAGES
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
except Exception:784],name='x')
y_ = tf.placeholder(tf.float32, [None, 8],name='labels')

#RESHAPE IMAGE IF NEED BE                                                     
x_image = tf.reshape(x, [-1,28,28,1])
#The number is the number of images that tensorboard will store
tf.summary.image('input',x_image,100)

#BUILD NETWORK
net = None
cmd = 'net=toy_%s.build(x_image,cfg.NUM_CLASS)' % cfg.ARCHITECTURE
exec(cmd)

#SOFTMAX
for i in range(cfg.NUM_CLASS):
  with tf.name_scope('softmax' + str(i)):
    softmax = tf.nn.softmax(logits=net)

#CROSS-ENTROPY                                                                 
  with tf.name_scope('cross_entropy' + str(i)):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=net))
    tf.summary.scalar('cross_entropy' + str(i), cross_entropy)
  a =[cross_entropy]
 #
  print(a)

#SUM OF ALL THE CROSS ENTROPIES
with tf.name_scope('Total_cross_entropy'):
  xent_total=tf.add_n(a)
#
  print(xent_total) 
  tf.summary.scalar('Total_Cross_Entropy', xent_total)

#TRAINING (RMS OR ADAM-OPTIMIZER OPTIONAL)                                     
with tf.name_scope('train'):
  train_step = tf.train.RMSPropOptimizer(0.0003).minimize(xent_total)

#ACCURACY                                                                      
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(y_,1))
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

    batch = make_images(cfg.TRAIN_BATCH_SIZE,debug=cfg.DEBUG)

    if i%100 == 0:
        
        s = sess.run(merged_summary, feed_dict={x:batch[0], y_:batch[1]})
        writer.add_summary(s,i)
        
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    
        print("step %d, training accuracy %g"%(i, train_accuracy))

    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})                                    

    if i%1000 ==0:
        batchtest = make_images(cfg.TEST_BATCH_SIZE,debug=cfg.DEBUG)
        test_accuracy = accuracy.eval(feed_dict={x:batchtest[0], y_:batchtest[1]})
        print("step %d, test accuracy %g"%(i, test_accuracy))

# post training test
batch = make_images(cfg.TEST_BATCH_SIZE,debug=cfg.DEBUG)
print("Final test accuracy %g"%accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))

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
fout.write('entry,label,prediction')
for idx in xrange(cfg.NUM_CLASS):
  fout.write(',score%02d' % idx)
fout.write('\n')

# run analysis
batch    = make_images(cfg.ANA_BATCH_SIZE,debug=cfg.DEBUG)
score_vv = softmax.eval(feed_dict={x: batch[0]})
for entry,score_v in enumerate(score_vv):
  label = int(np.argmax(batch[1][entry]))
  prediction = int(np.argmax(score_v))
  fout.write('%d,%d,%d' % (entry, label, prediction))
  for score in score_v:
    fout.write(',%g' % score)
  fout.write('\n')

fout.close()
