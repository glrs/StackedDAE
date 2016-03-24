from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from config import FLAGS
# from data import fill_feed_dict as fill_feed_dict
from utils import fill_feed_dict as fill_feed_dict

from sklearn.metrics import precision_score, confusion_matrix, classification_report
from sklearn.metrics import recall_score, f1_score, roc_curve, accuracy_score

from tools.visualize import plot_confusion_matrix as pcm
from tools.visualize import plot_roc_curve as roc

np.set_printoptions(linewidth=200)

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
    # the examples where the labels was in the top k (here k=1)
    # of all logits for that example.
    # correct: type = List (of booleans)
    correct = tf.nn.in_top_k(logits, labels, 1)
#     correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    
#     accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    y_p = tf.argmax(logits, 1)
#     l_p = tf.argmax(labels, 1)

    
    # Return the number of true entries. Cast because originally is bool.
    return tf.reduce_sum(tf.cast(correct, tf.int32)), correct, y_p


def do_eval(sess,
            eval_correct,
            predictions,
            examples_placeholder,
            labels_placeholder,
            label_map,
            data_set,
            title='Evaluation'):
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
    y_pred = []
    y_true = []
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    
    labels = tf.identity(labels_placeholder)
    
    for _ in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   examples_placeholder,
                                   labels_placeholder)
        corrects, y_prediction, y_trues = sess.run([eval_correct, predictions, labels], feed_dict=feed_dict)
        true_count += corrects
        y_pred += list(y_prediction)
        y_true += list(y_trues)
        
    accuracy = true_count / num_examples
    print(title + ' - Num examples: %d  Num correct: %d  Accuracy_score @ 1: %0.08f' %
          (num_examples, true_count, accuracy))

#     print("True output:", y_true)
#     print("Pred output:", y_pred)
    
    print("Precision:")
    print("\tNone: ", precision_score(y_true, y_pred, average=None, pos_label=None))
#     print("\tBinary:", precision_score(y_true, y_pred, average='binary'))
    print("\tMicro: %0.08f" % precision_score(y_true, y_pred, average='micro', pos_label=None))
    print("\tMacro: %0.08f" % precision_score(y_true, y_pred, average='macro', pos_label=None))
    print("\tWeighted: %0.08f" % precision_score(y_true, y_pred, average='weighted', pos_label=None))
#     print("\tSamples:", sklearn.metrics.precision_score(y_true, y_pred, average='samples'))
#     print("\tAccuracy_score: %0.08f" % accuracy_score(y_true, y_pred))
     
    print("Recall:")
    print("\tNone: ", recall_score(y_true, y_pred, average=None, pos_label=None))
#     print("\tBinary:", recall_score(y_true, y_pred, average='binary'))
    print("\tMicro: %0.08f" % recall_score(y_true, y_pred, average='micro', pos_label=None))
    print("\tMacro: %0.08f" % recall_score(y_true, y_pred, average='macro', pos_label=None))
    print("\tWeighted: %0.08f" % recall_score(y_true, y_pred, average='weighted', pos_label=None))
#     print("\tSamples:", sklearn.metrics.recall_score(y_true, y_pred, average='samples'))    
    
    print("F1_score:")
    print("\tNone: ", f1_score(y_true, y_pred, average=None, pos_label=None))
#     print("\tBinary:", f1_score(y_true, y_pred, average='binary'))
    print("\tMicro: %0.08f" % f1_score(y_true, y_pred, average='micro', pos_label=None))
    print("\tMacro: %0.08f" % f1_score(y_true, y_pred, average='macro', pos_label=None))
    print("\tWeighted: %0.08f" % f1_score(y_true, y_pred, average='weighted', pos_label=None))
#     print("\tSamples:", sklearn.metrics.f1_score(y_true, y_pred, average='samples'))

    print("True Length:", len(y_true))
    print("Prediction Length:", len(y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix")
    print(cm)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("\nNormalized confusion_matrix")
    print(cm_normalized)
    print("")
    
    print(classification_report(y_true, y_pred, target_names=label_map))

    pcm(cm, target_names=label_map, title=title)
    pcm(cm_normalized, target_names=label_map, title=title+"_Normalized")
    
    roc(y_pred, y_true, n_classes=len(label_map), title=title)
    
    print("\n=====================================================================================================\n")


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

  
