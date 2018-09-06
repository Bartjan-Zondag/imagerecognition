# Initialize necessary packages
import numpy as np 
import tensorflow as tf

import math 
import itertools
import progressbar
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.framework import ops

def one_hot_encoder(labels):
    """ Takes in a pandas series of labels and returns a one-hot encoded numpy array """

    # Convert to numpy array 
    labels = np.array(labels)

    # Initialize a label encoder
    label_encoder = LabelEncoder()

    # Label encode categorical variables
    labels = label_encoder.fit_transform(labels)

    # Save the class labels in a dictionary
    label_classes = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    # Define number of classes
    n_values = np.max(labels) + 1

    # Hot encode the classes
    hot_encoded = np.eye(n_values)[labels]

    return hot_encoded, label_classes


def create_placeholders(n_h0, n_w0, n_c0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_h0: scalar, height of an input image
    n_w0: scalar, width of an input image
    n_c0: scalar, number of channels of the input
    n_y : scalar, number of classes
        
    Returns:
    X_placeholder: placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    y_placeholder: placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X_placeholder = tf.placeholder(tf.float32, [None, n_h0, n_w0, n_c0])
    y_placeholder= tf.placeholder(tf.float32, [None, n_y])
    
    return X_placeholder, y_placeholder


def convolutional_layer(input_data, num_channels, num_filters, filter_shape, name):
    """ 
    Function used to prepare convolutional layers for image recognition training. 
    
    Arguments: 
    input_data   : input vector (beginning with X input vector and then subsequent layer)
    num_channels : the number of vectors corresponding to RGB values (=3)
    num_filters  : the number of filters to apply to convolve the pixel arrays
    filter_shape : the shape of the filters used to convolve the pixel arrays
    
    Returns:
    An Output Layer
    
    """
    # Define the filter shape for the convolutions (filter_shape x num_channels x num_filters)
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_channels, num_filters]
    
    # Initialize weights to assign to the filters 
    weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03), name=name+'_W')
    
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    output_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # Add the bias
    output_layer += bias

    # Apply a ReLU non-linear activation
    output_layer = tf.nn.relu(output_layer)

    return output_layer


def max_pool_layer(input_data, pool_shape):
    """ Perform Max Pooling """

    ksize = [1, pool_shape[0], pool_shape[1], 1]
    
    strides = [1, 2, 2, 1]
    
    output_layer = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding='SAME')
    
    return output_layer


# Define function for plotting confusion matrix 
def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=14, 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def random_batch(X, y, batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    batches -- list of synchronous (batch_X, batch_y)
    """
    
    m = X.shape[0]                  # number of training examples
    batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_y = y[permutation]

    # Step 2: Partition (shuffled_X, shuffled_y). Minus the end case.
    num_complete_minibatches = math.floor(m/batch_size) # number of mini batches of size batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        batch_X = shuffled_X[k * batch_size : k * batch_size + batch_size]
        batch_y = shuffled_y[k * batch_size : k * batch_size + batch_size]
        batch = (batch_X, batch_y)
        batches.append(batch)
    
    # Handling the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
        batch_X = shuffled_X[num_complete_minibatches * batch_size : m]
        batch_y = shuffled_y[num_complete_minibatches * batch_size : m]
        batch = (batch_X, batch_y)
        batches.append(batch)
    
    return batches


def batch_norm(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the 
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )


def batch_norm_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True,
                        )
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training is True:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, range(len(shape)-1))
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output