from __future__ import division

import os
import shutil
import sys
import time
import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from flags import FLAGS, home_out
from dae import DAE_Layer
from start_tensorboard import start
from data_handler import load_data

from utils import fill_feed_dict_dae
from utils import load_data_sets_pretraining
from utils import normalize_data, label_metadata
from visualize import hist_comparison

_data_dir = FLAGS.data_dir
_output_dir = FLAGS.output_dir
_summary_dir = FLAGS.summary_dir
_chkpt_dir = FLAGS.chkpt_dir

def _check_and_clean_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.mkdir(d)

loss_summaries = None

def main():
    home = home_out('')
    if not os.path.exists(home):
        os.makedirs(home)
    if not os.path.exists(_data_dir):
        os.mkdir(_data_dir)
        # os.makedirs also an option

        
    if not os.path.exists(_output_dir):
        os.mkdir(_output_dir)
    elif os.listdir(_output_dir):
        var = raw_input("Output folder is not empty. Clean it? (This will delete every file in it.) y/N: ")
        if var == 'y' or var == 'Y' or var == '1':
            _check_and_clean_dir(_output_dir)
        else:
            exit("Exiting... Please save your former output data and restart SDAE.")
    else:
        _check_and_clean_dir(_output_dir)

    _check_and_clean_dir(_summary_dir)
    _check_and_clean_dir(_chkpt_dir)
    
    os.mkdir(os.path.join(_chkpt_dir, '1'))
    os.mkdir(os.path.join(_chkpt_dir, '2'))
    os.mkdir(os.path.join(_chkpt_dir, '3'))
    os.mkdir(os.path.join(_chkpt_dir, 'fine_tuning'))
    
    start()
    
    start_time = time.time()
    datafile = load_data('RPKM', transpose=False)
    labelfile = load_data('Labels')
    print("Data Loaded. Duration:", time.time() - start_time)

    # Data Normalization
    datafile_norm = normalize_data(datafile, transpose=False)

    # Get data-sets (train, test) in a proper way
    data = load_data_sets_pretraining(datafile_norm, split_only=False)

    # Get Label Metadata
    mapped_labels, label_map = label_metadata(label_matrix=labelfile, label_col=7)
    num_classes = label_map.shape[0]
    
    nHLay = FLAGS.num_hidden_layers
    nHUnits = [getattr(FLAGS, "hidden{0}_units".format(j + 1)) for j in xrange(nHLay)]
    
    # Get the number of existed features (e.g. genes) in the data-set 
    num_features = datafile_norm.shape[1]
    # Create the shape of the AutoEncoder
    sdae_shape = [num_features] + nHUnits + [num_classes]
    
    with tf.Graph().as_default() as g:
        sess = tf.Session()

        y_all = {}
        for layer in xrange(3):
            y_all[layer] = []
            
            if layer == 0:
                x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, num_features), name='dae_input_from_layer_{0}'.format(layer))
                dae = DAE_Layer(in_data=x, prev_layer_size=num_features, next_layer_size=FLAGS.hidden1_units, nth_layer=layer+1, last_layer=False)
            elif layer == 1:
                x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, FLAGS.hidden1_units), name='dae_input_from_layer_{0}'.format(layer))
                dae = DAE_Layer(in_data=x, prev_layer_size=FLAGS.hidden1_units, next_layer_size=FLAGS.hidden2_units, nth_layer=layer+1, last_layer=False)
            else:
                x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, FLAGS.hidden2_units), name='dae_input_from_layer_{0}'.format(layer))
                dae = DAE_Layer(in_data=x, prev_layer_size=FLAGS.hidden2_units, next_layer_size=num_classes, nth_layer=layer+1, last_layer=False)# or True
            
            cost = dae.get_loss
            
            with tf.variable_scope("pretrain_{0}".format(layer+1)):
                train_op, global_step, loss_summaries = train(cost)
                
                summary_dir = pjoin(FLAGS.summary_dir, 'pretraining_{0}'.format(layer+1))
                summary_writer = tf.train.SummaryWriter(summary_dir, graph_def=sess.graph_def, flush_secs=FLAGS.flush_secs)
                summary_vars = [dae.get_w_b[0], dae.get_w_b[1]]
                        
                hist_summarries = [tf.histogram_summary(v.op.name, v) for v in summary_vars]
                hist_summarries.append(loss_summaries)
                summary_op = tf.merge_summary(hist_summarries)
            
                dae.vars_to_init.append(global_step)
                sess.run(tf.initialize_variables(dae.vars_to_init))
    
                print "| Layer | Epoch |   Cost   |   Step   |"
                print data.train.num_examples
                print data.all.num_examples
    
                for step in xrange(FLAGS.pretraining_epochs):# * data.train.num_examples):
    #                 for i in xrange(data.train.num_examples):
                        
                    feed_dict = fill_feed_dict_dae(data.train, x)
    
    #             if layer == 0:
                    c, _, y, z, w, b_in, b_out = sess.run([cost, train_op, dae.get_representation_y, dae.get_reconstruction_z, dae.get_w_b[0], dae.get_w_b[1], dae.get_b_recon], feed_dict=feed_dict)
    #             else:
    #                 c, _, gs, y, z, w, b_in, b_out = sess.run([cost, train_op, g_step, dae.get_representation_y, dae.get_reconstruction_z, dae.get_w_and_biases[0], dae.get_w_and_biases[1], dae.get_w_and_biases[2]], feed_dict=fill_feed_dict_dae(data.train, x))
    
                    if step % 1 == 0:
                        print '|  ', layer+1, '   |  ', step // data.train.num_examples + 1, '  | ', c, '  |     ', step, '     |'
                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                
        #         
        #         print np.asarray(y_all).shape
        #         print np.asarray(y_all[layer]).shape
            
            for _ in xrange(data.all.num_batches):
                feed_dict = fill_feed_dict_dae(data.all, x)
                
                y = sess.run(dae.get_representation_y, feed_dict=feed_dict)
                for j in xrange(np.asarray(y).shape[0]):
                    y_all[layer].append(y[j])

            print np.asarray(y_all[layer]).shape
            data = load_data_sets_pretraining(np.asarray(y_all[layer]), split_only=False)
            
        print "Finished..."

def train(layer, cost):
#    with tf.name_scope("Training"):
    # Add a scalar summary for the snapshot loss.
    loss_summaries = tf.scalar_summary(cost.op.name, cost)

    if layer is None:
        lr = FLAGS.supervised_learning_rate
    else:
        lr = layer._l_rate

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(lr)
    
    # Create a variable to track the global step.
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op, global_step, loss_summaries
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    