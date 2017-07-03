import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import toy_layers as L

def build(input_tensor, num_class=4):

    net = input_tensor
    # 1st conv layers ... default assumption: input 576x576
    net = L.conv2d(input_tensor=net, name='conv1_1', kernel=(3,3), stride=(2,2), num_filter=64, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv1_2', kernel=(3,3), stride=(1,1), num_filter=64, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool1",   kernel=(3,3), stride=(2,2))
    # 2nd conv layers ... 288 x 288
    net = L.conv2d(input_tensor=net, name='conv2_1', kernel=(3,3), stride=(1,1), num_filter=128, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv2_2', kernel=(3,3), stride=(1,1), num_filter=128, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool2",   kernel=(3,3), stride=(2,2))
    # 3rd conf layers ... 144 x 144
    net = L.conv2d(input_tensor=net, name='conv3_1', kernel=(3,3), stride=(1,1), num_filter=256, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv3_2', kernel=(3,3), stride=(1,1), num_filter=256, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool3",   kernel=(3,3), stride=(2,2))
    # 4rd conf layers ... 72 x 72
    net = L.conv2d(input_tensor=net, name='conv4_1', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv4_2', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool4",   kernel=(3,3), stride=(2,2))
    # 4rd conf layers ... 36 x 36
    net = L.conv2d(input_tensor=net, name='conv5_1', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv5_2', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool4",   kernel=(3,3), stride=(2,2))
    # by here it's 18x18
    return L.final_inner_product(input_tensor=net, name='fc_final', num_output=num_class)
# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,28,28,1])
    net = build(x)
    
