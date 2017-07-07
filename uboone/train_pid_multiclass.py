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

#START ACTIVE SESSION                                                         
sess = tf.InteractiveSession()

#PLACEHOLDERS                                                                 
x  = tf.placeholder(tf.float32, [None, image_dim[2] * image_dim[3]],name='x')
y_ = tf.placeholder(tf.float32, [None, cfg.NUM_CLASS],name='labels')

#RESHAPE IMAGE IF NEED BE                                                     
x_image = tf.reshape(x, [-1,image_dim[2],image_dim[3],1])
tf.summary.image('input',x_image,10)

#BUILD NETWORK
net = None
cmd = 'net=toy_%s.build(x_image,cfg.NUM_CLASS)' % cfg.ARCHITECTURE
exec(cmd)

#SOFTMAX
with tf.name_scope('sigmoid'):
  sigmoid = tf.nn.sigmoid(net)

#CROSS-ENTROPY                                                                
with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=net))
  tf.summary.scalar('cross_entropy',cross_entropy)

#TRAINING (RMS OR ADAM-OPTIMIZER OPTIONAL)                                    
with tf.name_scope('train'):
  train_step = tf.train.RMSPropOptimizer(0.0003).minimize(cross_entropy)

#ACCURACY                                                                     
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.rint(sigmoid), tf.rint(y_))
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

  data,label = proc.next()
  proc.read_next(cfg.TRAIN_BATCH_SIZE)

  loss,_ = sess.run([cross_entropy,train_step],feed_dict={x: data, y_: label})

  sys.stdout.write('Training in progress @ step %d loss %g\r' % (i,loss))
  sys.stdout.flush()

  if cfg.DEBUG:
    for idx in xrange(len(data)):
      img = None
      img = data[idx].reshape(image_dim[2],image_dim[3])
      
      adcpng = plt.imshow(img)
      imgname = 'debug_class_'
      for v in label[idx]:
        imgname += str(v)
      imgname += '_entry_%04d.png' % (i*cfg.TRAIN_ITERATIONS+idx)
      if os.path.isfile(imgname): raise Exception
      adcpng.write_png(imgname)
      plt.close()

      print '%-3d' % (i*cfg.TRAIN_ITERATIONS+idx),'...',
      print 'shape',img.shape,
      print img.min(),'=>', img.max(),'...',
      print img.mean(),'+/-',img.std(),'...',
      print 'max loc @',np.unravel_index(img.argmax(),img.shape),'...',
      print imgname
  
  if (i+1)%50 == 0:
    
    s = sess.run(merged_summary, feed_dict={x:data, y_:label})
    writer.add_summary(s,i)
  
    train_accuracy = sess.run(accuracy,feed_dict={x:data, y_: label})
    print
    print("step %d, training accuracy %g"%(i, train_accuracy))

  if (i+1)%200 == 0:
    save_path = saver.save(sess,'%s_step%06d' % (cfg.ARCHITECTURE,i))
    print 'saved @',save_path

#  if i%1000 ==0:
#    batchtest = make_images(cfg.TEST_BATCH_SIZE,debug=cfg.DEBUG,multiplicities=False)
#    test_accuracy = accuracy.eval(feed_dict={x:batchtest[0], y_:batchtest[1]})
#    print("step %d, test accuracy %g"%(i, test_accuracy))

# post training test
data,label = proc.next()
proc.read_next(cfg.TEST_BATCH_SIZE)
data,label = proc.next()
sess.run(accuracy,feed_dict={x:data[0],y_:label[1]})

print("Final test accuracy %g"%accuracy.eval(feed_dict={x: data, y_: label}))

# inform log directory
print('Run `tensorboard --logdir=%s` in terminal to see the results.' % cfg.LOGDIR)
