from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import FLAGS
# from data import fill_feed_dict as fill_feed_dict
from utils import fill_feed_dict as fill_feed_dict


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the labels is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    
    # Return the number of true entries. Cast because originally is bool.
    return tf.reduce_sum(tf.cast(correct, tf.int32)), correct


def do_eval(sess,
            eval_correct,
            examples_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        utils.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for _ in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   examples_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
    

def do_eval_summary(tag,
                    sess,
                    eval_correct,
                    examples_placeholder,
                    labels_placeholder,
                    data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for _ in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   examples_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    error = 1 - true_count / num_examples
    
    return sess.run(tf.scalar_summary(tag, tf.identity(error)))

  
