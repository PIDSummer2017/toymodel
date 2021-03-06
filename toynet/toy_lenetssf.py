import numpy as np
import tensorflow.python.platform
import tensorflow as tf
import toy_layers_ssf as L

def build(input_tensor, num_class=8):

    net = input_tensor
    # 1st conv layer
    net = L.conv2d(input_tensor=net, name='conv1_1', kernel=(3,3), stride=(1,1), num_filter=128, activation_fn=tf.nn.relu)
    # max pool
    net = L.avg_pool(input_tensor=net, name="pool1",   kernel=(2,2), stride=(2,2))
    # 2nd conv layer
    net = L.conv2d(input_tensor=net, name='conv2_1', kernel=(3,3), stride=(1,1), num_filter=128, activation_fn=tf.nn.relu)
    # max pool
    net = L.avg_pool(input_tensor=net, name="pool2",   kernel=(2,2), stride=(2,2))

    for i in range(num_class):
      z = L.final_inner_product(input_tensor=net, name='fc_final' + str(i), num_output=num_class))
      print(z)
    return z  
    
# script unit test
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [50,28,28,1])
    net = build(x)
