import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import toy_layers as L
import tensorflow.contrib.slim as slim

def build(input_tensor, num_class=4, trainable=True, reuse=False, keep_prob = 0.5):
    with tf.variable_scope("toy_pid", reuse=reuse):
        net = input_tensor
        num_outer_step = 5
        num_inner_step = 2
        num_base_filters = 32
        for outer_step in xrange(num_outer_step):
            for inner_step in xrange(num_inner_step):
                stride = 1
                activation_fn = tf.nn.relu
                #activation_fn = None
                if outer_step == 0 and inner_step == 0:
                    stride = 2
                if (outer_step+1) == num_outer_step and (inner_step+1) == num_inner_step:
                    activation_fn = None
                    #activation_fn = tf.nn.relu
                net = slim.conv2d (net, 
                                   num_outputs=num_base_filters * (outer_step+2), 
                                   kernel_size=3, 
                                   stride=stride, 
                                   normalizer_fn = slim.batch_norm, 
                                   #normalizer_fn = None, 
                                   activation_fn = activation_fn,
                                   trainable = trainable, 
                                   #activation_fn = None,
                                   weights_regularizer=slim.l2_regularizer(0.001),
                                   biases_regularizer=slim.l2_regularizer(0.001),
                                   biases_initializer=tf.constant_initializer(0.1),
                                   scope='conv%d_%d' % (outer_step,inner_step) )
                print('Step {:d}.{:d} ... {:s}'.format(outer_step,inner_step,net.shape))
               
            #net = slim.max_pool2d (net, 2, scope='maxpool%d' % outer_step)
            net = slim.avg_pool2d (net, 2, scope='avgpool%d' % outer_step)
            print('Pool {:d} ... {:s}'.format(outer_step,net.shape))
            
        net = slim.flatten(net, scope='flatten')
        net = slim.dropout(net,keep_prob=keep_prob, scope='dropout')
        print('Flattened ... {:s}'.format(net.shape))
        '''
        for x in xrange(3):
            if x == 0 : output = 1024
            if x == 1 : output = 256
            if x == 2 : output = 5
            activation_fn = None
            net = slim.fully_connected (net, output, activation_fn=activation_fn, biases_initializer=tf.constant_initializer(0.1), scope='fc%i'%x )
            with tf.variable_scope('drop_out%i'%x):
                net = slim.dropout (net, keep_prob,      scope='dropout%i'%x)
            print('FC{:n} ... {:s}'.format(x, net.shape))
        '''
        activation_fn = None
        output=5
        net = slim.fully_connected (net, output, activation_fn=activation_fn, biases_initializer=tf.constant_initializer(0.1), scope='fc0' )
    return net

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,512,512,1])
    net = build(x)
    
