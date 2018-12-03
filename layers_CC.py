import tensorflow as tf


def all_views_conv_layer(input_layer, layer_name, number_of_filters=32, filter_size=(3, 3), stride=(1, 1),
                         padding='VALID', biases_initializer=tf.zeros_initializer()):
    """Convolutional layers for 2x2 views input 4-DCN"""
    input_r_cc = input_layer

    # with tf.variable_scope(layer_name + "_CC") as cc_cope:
    h_r_cc = tf.contrib.layers.convolution2d(inputs=input_r_cc, num_outputs=number_of_filters,
                                             kernel_size=filter_size, stride=stride, padding=padding, reuse=False,
                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                                             biases_initializer=biases_initializer)


    h = (h_r_cc)

    return h


def all_views_max_pool(input_layer, stride=(2, 2)):
    """Max-pool across all 4 views"""

    input_r_cc = input_layer
    output_r_cc = tf.nn.max_pool(input_r_cc, ksize=[1, stride[0], stride[1], 1], strides=[1, stride[0], stride[1], 1],
                                     padding='SAME')
    output = (output_r_cc)

    return output


def all_views_global_avg_pool(input_layer):
    """Average-pool across all 4 views"""

    input_r_cc = input_layer
    input_layer_shape = input_r_cc.get_shape()
    pooling_shape = [1, input_layer_shape[1], input_layer_shape[2], 1]

    output_r_cc = tf.nn.avg_pool(input_r_cc, ksize=pooling_shape, strides=pooling_shape, padding='SAME')
    output = (output_r_cc)
    return output

def all_views_flattening_layer(input_layer):
    """Flatten and concatenate all activations from all 4 views"""


    input_r_cc = input_layer
    input_layer_shape = input_r_cc.get_shape()
    input_layer_size = int(input_layer_shape[1]) * int(input_layer_shape[2]) * int(input_layer_shape[3])

    h_r_cc_flat = tf.reshape(input_r_cc, [-1, input_layer_size])

    h_cc_flat = tf.concat(axis=1, values=[h_r_cc_flat])

    h_flat = (h_cc_flat)
    return h_flat

def fc_layer(input_layer, number_of_units=128, activation_fn=tf.nn.relu, reuse=None, scope=None):
    """Fully connected layer"""

    input_r_cc = input_layer
    h_cc = tf.contrib.layers.fully_connected(inputs=input_r_cc, num_outputs=number_of_units, activation_fn=activation_fn,
                                              reuse=reuse, scope=scope)
    h = (h_cc)
    return h

def softmax_layer(input_layer, number_of_outputs=3):
    """Softmax layer"""
    input_layer_cc = input_layer
        # with tf.variable_scope('fully_connected_1') as fully_scope:
    y_prediction_cc = tf.contrib.layers.fully_connected(inputs=input_layer_cc, num_outputs=number_of_outputs,
                                                            activation_fn=tf.nn.softmax)
    y_prediction = (y_prediction_cc)
    return y_prediction

def dropout_layer(input_layer, nodropout_probability=0.50):
    """Dropout layer"""
    input_layer_cc= input_layer
    output_cc = tf.nn.dropout(input_layer_cc, nodropout_probability)

    output = (output_cc)

    return output


def gaussian_noise_layer(input_layer, std):
    """Additive gaussian noise layer"""

    noise = tf.random_normal(tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)

    output = tf.add_n([input_layer, noise])

    return output


def all_views_gaussian_noise_layer(input_layer, std):
    """Add gaussian noise across all 4 views"""

    input_r_cc = input_layer

    output_r_cc = gaussian_noise_layer(input_r_cc, std)


    output = (output_r_cc)

    return output
