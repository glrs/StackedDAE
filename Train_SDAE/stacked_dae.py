from __future__ import division

import tensorflow as tf
import numpy as np
import time
import sklearn

from sklearn.metrics import precision_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score, roc_curve

from dae import DAE_Layer
from os.path import join as pjoin

#from utils import load_data_sets_pretraining, write_csv
from tools.utils import fill_feed_dict, fill_feed_dict_dae
from tools.evaluate import do_eval_summary, evaluation, do_eval
from tools.config import FLAGS
from tools.visualize import make_heatmap
from tensorflow.python.framework.errors import FailedPreconditionError


class Stacked_DAE(object):
    
    def __init__(self, net_shape, session=None, selfish_layers=False):
        """ Stack De-noising Autoencoder (SDAE) initialization 
        
        Args:
            net_shape: The network architecture of the SDAE
            session : The tensorflow session
            selfish_layers: Whether the layers are going to be trained individually
                            or dependent to the direct output of the previous layer
                            (Theoretically: using it is faster, but memory costly)
        Tips:
            Using selfish_layers needs some extra handling.
              * Feed each individual De-noising Autoencoder (DAE) directly.
                    (e.g. feed_dict = {sdae.get_layers[i]._x : input_data})
              * Reassign/Reload the input data-set with the data-set for the next
                layer, obtained by using the genrate_next_dataset() function.
                    (e.g. in this case load_data_sets_pretraining(next_dataset, split_only=False))
        """
        self._sess = session
        self._net_shape = net_shape
        self.nHLayers = len(self._net_shape) - 2
        self._selfish_layers = selfish_layers
        self.loss_summaries = None
        
        if self._selfish_layers:
            self._x = None
            self._y_dataset = {}
        else:
            self._x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, self._net_shape[0]), name='dae_input_layer')

        self._dae_layers = []
        self._weights = []
        self._biases = []
        self.weights = []
        self.biases = []
        self._create_network()

    def _create_network(self):
        is_last_layer = False
        for layer in xrange(self.nHLayers + 1):
            with tf.name_scope("Layer_{0}".format(layer)):
                if self._selfish_layers: 
                    x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, self._net_shape[layer]), name='dae_input_from_layer_{0}'.format(layer))
                    self._y_dataset[layer] = []
                else:
                    if layer == 0:
                        x = self._x

                    else:
                        x = self._dae_layers[layer-1].clean_activation()
#                         x = self._dae_layers[layer-1].get_representation_y

                if not layer < self.nHLayers:
                    is_last_layer = True
#                 if layer == self.nHLayers:
#                     break

                dae_layer = DAE_Layer(in_data=x, prev_layer_size=self._net_shape[layer],
                                      next_layer_size=self._net_shape[layer+1], nth_layer=layer+1,
                                      last_layer=is_last_layer)
                
                self._dae_layers.append(dae_layer)

    @property
    def session(self):
        return self._sess

    @property
    def get_layers(self):
        return self._dae_layers
    
    @property
    def get_weights(self):
#         if len(self.weights) != self.nHLayers + 1:
        self.weights = []
        for n in xrange(self.nHLayers + 1):
            if self.get_layers[n].get_w:
                try:
                    self.weights.append(self._sess.run(self.get_layers[n].get_w))
                except FailedPreconditionError:
                    break
            else:
                break

        return self.weights

    @property
    def get_biases(self):
#         if len(self.biases) != self.nHLayers + 1:
        self.biases = []
        for n in xrange(self.nHLayers + 1):
            if self.get_layers[n].get_b:
                try:
                    self.biases.append(self._sess.run(self.get_layers[n].get_b))
                except FailedPreconditionError:
                    break
            else:
                break

        return self.biases
    
    def get_activation(self, x, layer, use_fixed=True):
        return self._sess.run(self.get_layers[layer].clean_activation(x_in=x, use_fixed=use_fixed))
#         return self._sess.run(tf.sigmoid(tf.nn.bias_add(tf.matmul(x, self.get_weights[layer]), self.get_biases[layer]), name='activate'))

    def train(self, cost, layer=None):
#         with tf.name_scope("Training"):
        # Add a scalar summary for the snapshot loss.
        self.loss_summaries = tf.scalar_summary(cost.op.name, cost)

        if layer is None:
            lr = FLAGS.supervised_learning_rate
        else:
            lr = self.get_layers[layer]._l_rate

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(lr)
        
        # Create a variable to track the global step.
        global_step = tf.Variable(0, trainable=False, name='global_step')
        
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op, global_step

    def calc_last_x(self, X):
        tmp = X
        for layer in self.get_layers:
            tmp = layer.clean_activation(x_in=tmp, use_fixed=False)
#         print(tmp, self._net_shape[-2], self._net_shape[-1])
#         dae_layer = DAE_Layer(in_data=tmp, prev_layer_size=self._net_shape[-2],
#                                       next_layer_size=self._net_shape[-1], nth_layer=len(self._net_shape)-1,
#                                       last_layer=True)
# 
#         self._dae_layers.append(dae_layer)
#         tmp = self.get_layers[-1].clean_activation(x_in=tmp, use_fixed=False)
        
        return tmp

    def add_final_layer(self, input_x):
        last_x = self.calc_last_x(input_x)
        print "Last layer added:", last_x.get_shape()
        return last_x
    
#     def finetune_net(self):
#         last_output = self._x
#         
#         for layer in xrange(self.nHLayers + 1):
#             w = self.get_layers[layer]
    
    def genrate_next_dataset(self, from_dataset, layer):
        """ Generate next data-set
        Note: This function has a meaning only if selfish layers are in use.
        It takes as input the data-set and transforms it using the previously
        trained layer in order to obtain it's output. The output of that layer
        is saved as a data-set to be used as input for the next one.
        
        Args:
            from_dataset: The data-set you want to transform (usually
                            the one that the previous layer is trained on)
            layer : The layer to be used for the data transformation
        Returns:
            numpy array: The new data-set to be used for the next layer
        """
        if self._selfish_layers:
            for _ in xrange(from_dataset.num_batches):
                feed_dict = fill_feed_dict_dae(from_dataset, self.get_layers[layer]._x)

                y = self._sess.run(self.get_layers[layer].clean_activation(), feed_dict=feed_dict)
                for j in xrange(np.asarray(y).shape[0]):
                    self._y_dataset[layer].append(y[j])
                    
            return np.asarray(self._y_dataset[layer])
        else:
            print "Note: This function has a meaning only if selfish layers are in use."
            return None

def pretrain_sdae(input_x, shape):
    with tf.Graph().as_default():# as g:
        sess = tf.Session()
        
        sdae = Stacked_DAE(net_shape=shape, session=sess, selfish_layers=False)

        for layer in sdae.get_layers[:-1]:
            with tf.variable_scope("pretrain_{0}".format(layer.which)):
                cost = layer.get_loss
                train_op, global_step = sdae.train(cost, layer=layer.which)

                summary_dir = pjoin(FLAGS.summary_dir, 'pretraining_{0}'.format(layer.which))
                summary_writer = tf.train.SummaryWriter(summary_dir, graph_def=sess.graph_def, flush_secs=FLAGS.flush_secs)
                summary_vars = [layer.get_w_b[0], layer.get_w_b[1]]
                        
                hist_summarries = [tf.histogram_summary(v.op.name, v) for v in summary_vars]
                hist_summarries.append(sdae.loss_summaries)
                summary_op = tf.merge_summary(hist_summarries)

                '''
                 You can get all the trainable variables using tf.trainable_variables(),
                 and exclude the variables which should be restored from the pretrained model.
                 Then you can initialize the other variables.
                '''

                layer.vars_to_init.append(global_step)
                sess.run(tf.initialize_variables(layer.vars_to_init))

                print("\n\n")
                print "|  Layer   |   Epoch    |   Step   |    Loss    |"
                
                for step in xrange(FLAGS.pretraining_epochs * input_x.train.num_examples):
                    feed_dict = fill_feed_dict_dae(input_x.train, sdae._x)
    
                    loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
                    
                    if step % 1000 == 0:
                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        
                        output = "| Layer {0}  | Epoch {1}    |  {2:>6}  | {3:10.4f} |"\
                                     .format(layer.which, step // input_x.train.num_examples + 1, step, loss)
                        print output
    
                # Note: Use this style if you are using the shelfish_layer choice.
                # This way you keep the activated data to be fed to the next layer.
                # next_dataset = sdae.genrate_next_dataset(from_dataset=input_x.all, layer=layer.which)
                # input_x = load_data_sets_pretraining(next_dataset, split_only=False)

        # Save Weights and Biases for all layers
        for n in xrange(len(shape) - 2):
            w = sdae.get_layers[n].get_w
            b = sdae.get_layers[n].get_b
            W, B = sess.run([w, b])

            np.savetxt(pjoin(FLAGS.output_dir, 'Layer_' + str(n) + '_Weights.txt'), np.asarray(W), delimiter='\t')
            np.savetxt(pjoin(FLAGS.output_dir, 'Layer_' + str(n) + '_Biases.txt'), np.asarray(B), delimiter='\t')
            make_heatmap(W, 'weights_'+ str(n))

    print "\nPretraining Finished...\n"
    return sdae



def finetune_sdae(sdae, input_x, n_classes, label_map):
    print "Starting Fine-tuning..."
    with sdae.session.graph.as_default():
        sess = sdae.session
        
        n_features = sdae._net_shape[0]
        
        x_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, n_features), name='input_pl')
        labels_pl = tf.placeholder(tf.int32, shape=FLAGS.batch_size, name='labels_pl')
        labels = tf.identity(labels_pl)
        
        # Get the supervised fine tuning net
        logits = sdae.add_final_layer(x_pl)
#         logits = sdae.finetune_net(input_x)
        loss = loss_supervised(logits, labels_pl, n_classes)

        train_op, _ = sdae.train(loss)
        eval_correct, corr, y_pred = evaluation(logits, labels_pl)
        
        hist_summaries = [layer.get_w for layer in sdae.get_layers]
        hist_summaries.extend([layer.get_b for layer in sdae.get_layers])
        
        hist_summaries = [tf.histogram_summary(v.op.name + "_fine_tuning", v) for v in hist_summaries]
        
        summary_op = tf.merge_summary(hist_summaries)
        
        summary_writer = tf.train.SummaryWriter(pjoin(FLAGS.summary_dir, 'fine_tuning'),
                                                graph_def=sess.graph_def,
                                                flush_secs=FLAGS.flush_secs)
        
        sess.run(tf.initialize_all_variables())
        
        steps = FLAGS.finetuning_epochs * input_x.train.num_examples
        for step in xrange(steps):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(input_x.train, x_pl, labels_pl)
            
            _, loss_value, ev_corr, c, y_true = sess.run([train_op, loss, eval_correct, corr, labels], feed_dict=feed_dict)
            
            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 1000 == 0:
                # Print status to stdout.
                print "\nLoss: ", loss_value
#                 print "Eval corr:", ev_corr
#                 print "Correct:", c
#                 print "Y_pred:", y_pred
#                 print "Label_pred:", y_true
                
#                 y_true = np.argmax(labels_pl, 0)
                
                
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                
                print 'Evaluation Sum:', ev_corr, '/', len(c)
#                 print('Evaluation Corrects:', eval_corr)
#                 print('Logits:', lgts)
                print "---------------"
                
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if (step + 1) % 1000 == 0 or (step + 1) == steps:
                train_sum = do_eval_summary("training_error",
                                            sess,
                                            eval_correct,
                                            x_pl,
                                            labels_pl,
                                            input_x.train)
                
                if input_x.validation is not None:
                    val_sum = do_eval_summary("validation_error",
                                              sess,
                                              eval_correct,
                                              x_pl,
                                              labels_pl,
                                              input_x.validation)
                
                test_sum = do_eval_summary("test_error",
                                           sess,
                                           eval_correct,
                                           x_pl,
                                           labels_pl,
                                           input_x.test)
                
                summary_writer.add_summary(train_sum, step)
                if input_x.validation is not None:
                    summary_writer.add_summary(val_sum, step)
                summary_writer.add_summary(test_sum, step)

        for n in xrange(len(sdae._net_shape) - 1):
            w = sdae.get_layers[n].get_w
            b = sdae.get_layers[n].get_b
            W, B = sess.run([w, b])

            np.savetxt(pjoin(FLAGS.output_dir, 'Finetuned_Layer_' + str(n) + '_Weights.txt'), np.asarray(W), delimiter='\t')
            np.savetxt(pjoin(FLAGS.output_dir, 'Finetuned_Layer_' + str(n) + '_Biases.txt'), np.asarray(B), delimiter='\t')
            make_heatmap(W, 'Finetuned_weights_'+ str(n))

        do_eval(sess, eval_correct, y_pred, x_pl, labels_pl, label_map, input_x.train, title='Final_Train')
        do_eval(sess, eval_correct, y_pred, x_pl, labels_pl, label_map, input_x.test, title='Final_Test')
        if input_x.validation is not None:
            do_eval(sess, eval_correct, y_pred, x_pl, labels_pl, label_map, input_x.validation, title='Final_Validation')
        
    print "Fine-tuning Finished..."
    return sdae


def loss_supervised(logits, labels, num_classes):
    """Calculates the loss from the logits and the labels.
    
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    
    Returns:
      loss: Loss tensor of type float.
    """
    
    # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
    # to 1-hot dense float vectors (that is we will have batch_size vectors,
    # each with NUM_CLASSES values, all of which are 0.0 except there will
    # be a 1.0 in the entry corresponding to the label).
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
    
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss
