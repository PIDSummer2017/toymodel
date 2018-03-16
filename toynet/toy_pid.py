import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import toy_layers as L
import tensorflow.contrib.slim as slim

def build(input_tensor, num_class=5, trainable=True, reuse=False):

    with tf.variable_scope("toy_pid",reuse=reuse):
        net = input_tensor
        num_outer_step = 5
        num_inner_step = 2
        num_base_filters = 32
        for outer_step in xrange(num_outer_step):
            for inner_step in xrange(num_inner_step):
                stride = 1
                activation_fn = tf.nn.relu
                if outer_step == 0 and inner_step == 0:
                    stride = 2
                if (outer_step+1) == num_outer_step and (inner_step+1) == num_inner_step:
                    activation_fn = None
                net = slim.conv2d (net, 
                                   num_outputs=num_base_filters * (outer_step+1), 
                                   kernel_size=3, 
                                   stride=stride, 
                                   normalizer_fn = slim.batch_norm, 
                                   activation_fn = activation_fn,
                                   scope='conv%d_%d' % (outer_step,inner_step) )
                print('Step {:d}.{:d} ... {:s}'.format(outer_step,inner_step,net.shape))

            net = slim.max_pool2d (net, 2, scope='maxpool%d' % outer_step)
            print('Pool {:d} ... {:s}'.format(outer_step,net.shape))
        net = slim.flatten(net, scope='flatten')
        print('Flattened ... {:s}'.format(net.shape))
        net = slim.fully_connected (net, 1024,      scope='fc0'    )
        print('FC1 ... {:s}'.format(net.shape))
        net = slim.fully_connected (net, 256,      scope='fc1'    )
        print('FC2 ... {:s}'.format(net.shape))
        net = slim.fully_connected (net, num_class, scope='fc2'    )
        print('FC3 ... {:s}'.format(net.shape))
    return net
    #Final Array will be 41472 x 5 (9x9x512 and 5 classes)

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,512,512,1])
    net = build(x)
    
