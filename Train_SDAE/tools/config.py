from __future__ import division
import os
from os.path import join as pjoin

import sys

import tensorflow as tf

WEB_OUT = '/var/www/html/'

def home_out(path):
    return pjoin(os.environ['HOME'], 'tmp_StackedDAE', 'Allan', path)

def web_out(path):
    # Just a quick manual flag for changes between local and remote VMs
    if False:
        return pjoin(WEB_OUT, 'StackedDAE', path)
    else:
        return home_out(path)


flags = tf.app.flags
FLAGS = flags.FLAGS

# Autoencoder Architecture Specific Flags
flags.DEFINE_integer('num_hidden_layers', 3, 'Number of hidden layers')

flags.DEFINE_integer('hidden1_units', 25, 'Number of units in hidden layer 1.')   # 2000
flags.DEFINE_integer('hidden2_units', 15, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3_units', 10, 'Number of units in hidden layer 3.')

# flags.DEFINE_integer('example_features', EXAMPLE_FEATURES, 'Total number of features (genes)')  # image_pixels
# flags.DEFINE_integer('num_classes', 10, 'Number of classes')

flags.DEFINE_float('unsupervised_learning_rate', 0.0001, 'Unsupervised initial learning rate.')
flags.DEFINE_float('supervised_learning_rate', 0.01, 'Supervised initial learning rate.')

flags.DEFINE_float('pre_layer1_learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('pre_layer2_learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('pre_layer3_learning_rate', 0.0001, 'Initial learning rate.')

flags.DEFINE_boolean('emphasis', False, 'Whether to use Emphasis or Not')
flags.DEFINE_string('emphasis_type', 'Double', 'Type of Emphasis for the Cross Entropy. [Double, Full]')

flags.DEFINE_float('default_noise', [0.0, 'MN'], 'Default Noise ratio and type to apply on the data')

# flags.DEFINE_float('noise_1', [0.50, 'MN'], 'Noise ratio to apply on the data, and the type of noise')
# flags.DEFINE_float('noise_2', [0.50, 'MN'], 'Noise ratio to apply on the data, and the type of noise')
# flags.DEFINE_float('noise_3', [0.50, 'MN'], 'Noise ratio to apply on the data, and the type of noise')

flags.DEFINE_float('noise_1', [0.20, 'MN'], 'Noise ratio to apply on the data, and the type of noise')
flags.DEFINE_float('noise_2', [0.20, 'MN'], 'Noise ratio to apply on the data, and the type of noise')
flags.DEFINE_float('noise_3', [0.20, 'MN'], 'Noise ratio to apply on the data, and the type of noise')

""" TODO: ADD a flag for activation function (sigmoid, tanh, etc.) """

# Constants
# flags.DEFINE_integer('seed', 1234, 'Random seed')

flags.DEFINE_integer('batch_size', 9, 'Batch size. Must divide evenly into the dataset sizes.')   # 100

flags.DEFINE_integer('pretraining_epochs', 20, 'Number of training epochs for pretraining layers')  # 60
flags.DEFINE_integer('finetuning_epochs', 60, 'Number of training epochs for fine tuning supervised step')

flags.DEFINE_float('zero_bound', 1.0e-9, 'Value to use as buffer to avoid numerical issues at 0')
flags.DEFINE_float('one_bound', 1.0 - 1.0e-9, 'Value to use as buffer to avoid numerical issues at 1')

flags.DEFINE_float('flush_secs', 120, 'Number of seconds to flush summaries')   # 120

# Directories
flags.DEFINE_string('data_dir', home_out('data'), 'Directory to put the training data.')

flags.DEFINE_string('output_dir', web_out('output'), 'Directory to put the output data.')

flags.DEFINE_string('summary_dir', home_out('summaries'), 'Directory to put the summary data')

flags.DEFINE_string('chkpt_dir', home_out('chkpts'), 'Directory to put the model checkpoints')

# TensorBoard
# flags.DEFINE_boolean('no_browser', True, 'Whether to start browser for TensorBoard')

# Python
flags.DEFINE_string('python', sys.executable, 'Path to python executable')


