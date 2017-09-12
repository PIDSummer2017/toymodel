import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import toy_layers as L

def build(input_tensor, num_class=4):

    net = input_tensor
    # 1st conv layers ... default assumption: input 576 x 576 x 1
    net = L.conv2d(input_tensor=net, name='conv1_1', kernel=(3,3), stride=(2,2), num_filter=64, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv1_2', kernel=(3,3), stride=(1,1), num_filter=64, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool1",   kernel=(3,3), stride=(2,2))
    # 2nd conv layers ... 144 x 144 x 64 (because of stride 2 in conv1_1)
    net = L.conv2d(input_tensor=net, name='conv2_1', kernel=(3,3), stride=(1,1), num_filter=128, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv2_2', kernel=(3,3), stride=(1,1), num_filter=128, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool2",   kernel=(3,3), stride=(2,2))
    # 3rd conf layers ... 72 x 72 x 128
    net = L.conv2d(input_tensor=net, name='conv3_1', kernel=(3,3), stride=(1,1), num_filter=256, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv3_2', kernel=(3,3), stride=(1,1), num_filter=256, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool3",   kernel=(3,3), stride=(2,2))
    # 4rd conf layers ... 36 x 36 x 256
    net = L.conv2d(input_tensor=net, name='conv4_1', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv4_2', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool4",   kernel=(3,3), stride=(2,2))
    # 4rd conf layers ... 18 x 18 x 512
    net = L.conv2d(input_tensor=net, name='conv5_1', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    net = L.conv2d(input_tensor=net, name='conv5_2', kernel=(3,3), stride=(1,1), num_filter=512, activation_fn=tf.nn.relu)
    # max pool
    net = L.max_pool (input_tensor=net, name="pool5",   kernel=(3,3), stride=(2,2))
    # by here it's 9 x 9 x 512
    return L.final_inner_product(input_tensor=net, name='fc_final', num_output=num_class)
    #Final Array will be 41472 x 5 (9x9x512 and 5 classes)

# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,28,28,1])
    net = build(x)
    
