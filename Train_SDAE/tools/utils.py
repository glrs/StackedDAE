import numpy as np
import csv
from config import FLAGS

from sklearn.cross_validation import train_test_split

class DataSet(object):
    def __init__(self, examples, labels=None):
        if labels is not None:
            assert len(examples) == len(labels), (
                                        'examples.shape: %s labels.shape: %s'
                                        % (examples.shape, labels.shape))

        self._num_examples = examples.shape[0]
        self._examples = examples
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def examples(self):
        return self._examples
    
    @property
    def labels(self):
        return self._labels
     
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    @property
    def index_in_epoch(self):
        return self._index_in_epoch
        
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            
            self._examples = self._examples[perm]

            if self._labels is not None:
                self._labels = self._labels[perm]
    
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
            
        end = self._index_in_epoch
        
        if self._labels is None:
            return self._examples[start:end] #self._examples.iloc[start:end]
        else:
#             return self._examples.iloc[start:end], self._labels.iloc[start:end]
            return self._examples[start:end], self._labels[start:end]
        

class DataSetPreTraining(object):
    def __init__(self, examples):
        self._num_examples = examples.shape[0]
        self._examples = examples

        self._examples[self._examples < FLAGS.zero_bound] = FLAGS.zero_bound
        self._examples[self._examples > FLAGS.one_bound] = FLAGS.one_bound

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def examples(self):
        return self._examples
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def num_batches(self):
        return self.num_examples / FLAGS.batch_size
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    @property
    def index_in_epoch(self):
        return self._index_in_epoch

#     """ TODO: Under implementation """
#     def same_batch(self):
#         pass

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._examples[perm]
            
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

#             print self._num_examples
            assert batch_size <= self._num_examples
            
        end = self._index_in_epoch
        
        return self._examples[start:end]


def load_data_sets(input_data, labels, split_only=True, valid_set=False):
    class DataSets(object):
        pass
    data_sets = DataSets()

    print("\nSplitting to Train & Test sets for Finetuning")

    if valid_set:
        train_examples, test_examples, train_labels, test_labels = \
                        train_test_split(input_data, labels, test_size=0.2)
        train_examples, validation_examples, train_labels, validation_labels = \
                        train_test_split(train_examples, train_labels, test_size=0.05)
        data_sets.validation = DataSet(validation_examples, validation_labels)
    else:
        train_examples, test_examples, train_labels, test_labels = \
                        train_test_split(input_data, labels, test_size=0.3)
        data_sets.validation = None

#     validation_examples = input_data[:VALIDATION_SIZE]
#     train_examples = input_data[VALIDATION_SIZE:]

    data_sets.train = DataSet(train_examples, train_labels)
    data_sets.test = DataSet(test_examples, test_labels)
    
    if not split_only:
        data_sets.all = DataSet(input_data, labels)
    
    return data_sets



def load_data_sets_pretraining(input_data, split_only=True, valid_set=False):
    """ Load data-sets for pre-training
    Data-sets for pre-training does not include labels. It takes
    an input data-set and it splits it in train, test and validation
    (optional) sets. Then it returns these subsets as DataSetPreTraining
    objects which have the ability to give the data in batches (among
    other useful functions). If split_only argument is False then it
    also returns the whole input data-set as a DataSetPreTraining object.
    
    Args:
        input_data: The data-set to be split.
        split_only: If True it just splits the data-set and returns its
                    subsets as DataSetPreTraining objects, otherwise it
                    also returns the data-set as DataSetPreTraining object.
        valid_set:  Whether to create a validation set along with test
                    and train or not (default False)
    """
    class DataSets(object):
        pass
    data_sets = DataSets()
    
    print("\nSplitting to Train & Test sets for pre-training")
    
    if valid_set:
        train_examples, test_examples = train_test_split(input_data, test_size=0.20)
        train_examples, validation_examples = train_test_split(train_examples, test_size=0.05)
        data_sets.validation = DataSetPreTraining(validation_examples)
    else:
        train_examples, test_examples = train_test_split(input_data, test_size=0.3)
        data_sets.validation = None

    if not split_only:
        data_sets.all = DataSetPreTraining(input_data)

    data_sets.train = DataSetPreTraining(train_examples)
    data_sets.test = DataSetPreTraining(test_examples)

    return data_sets


'''
""" TODO: ADD more noise functions such as Gaussian noise etc. """
def _add_noise(x, ratio, n_type='MN'):
    """ Noise adding (or input corruption)
    This function adds noise to the given dataset.
    
    Args:
        x    : The input dataset for the noise to be applied (numpy array)
        ratio: The percentage of the data affected by the noise addition
        n_type: The type of noise to be applied.
                Choices: MN (masking noise), SP (salt-and-pepper noise)
    """
'''

def fill_feed_dict_dae(data_set, input_pl, batch_size=None):
    b_size = FLAGS.batch_size if batch_size is None else batch_size

    input_feed = data_set.next_batch(b_size)
    feed_dict = { input_pl: input_feed }

    return feed_dict


def fill_feed_dict(data_set, input_pl, labels_pl, batch_size=None):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The examples placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    b_size = FLAGS.batch_size if batch_size is None else batch_size
    
    examples_feed, labels_feed = data_set.next_batch(b_size)

    feed_dict = {
        input_pl: examples_feed,
        labels_pl: labels_feed
    }

    return feed_dict


def normalize_data(x, transpose=False):
    # Normalization across the whole matrix
#     x_max = np.max(x)
#     x_min = np.min(x)
#     x_norm = (x - x_min) / np.float32(x_max - x_min)
    
    
    # Normalization across the features
    x_norm = []
    if transpose:
        x = np.transpose(x)
        print("\nData Transposed.")

    print "\nNormalizing", len(x), "Features..."
    for i in range(len(x)):
        x_norm.append((x[i] - np.min(x[i])) / np.float32(np.max(x[i]) - np.min(x[i])))
        if np.isnan(x_norm[i]).any():
            print("NAN at:", i)

    """ OR  (norm='l1' or 'l2' or 'max')
    from sklearn.preprocessing import normalize
    x_norm = normalize(input_data, axis=??, norm='??')
    """
    print("Normalization: Done. Transposing...")
    return np.asarray(np.transpose(x_norm))


def label_metadata(label_matrix, label_col):
    # Check whether the column value is given as index (number) or name (string) 
    try:
        label_col = int(label_col)
        
        # If given as number, take the name of the column out of it
        label_col = label_matrix.columns[label_col]
    except ValueError:
        pass
    
    import pandas as pd
    # Get the unique classes in the given column, and how many of them are there
    unique_classes = pd.unique(label_matrix[label_col].ravel())
    #num_classes = unique_classes.shape[0]
    
    # Map the unique n classes with a number from 0 to n  
    label_map = pd.DataFrame({label_col: unique_classes, label_col+'_id':range(len(unique_classes))})
    
    # Replace the given column's values with the mapped equivalent
    mapped_labels = label_matrix.replace(label_map[[0]].values.tolist(), label_map[[1]].values.tolist())
    
    # Return the mapped labels as numpy list and the label map (unique classes and number can be obtained from map)
    return np.reshape(mapped_labels[[label_col]].values, (mapped_labels.shape[0],)), np.asarray(label_map) #, unique_classes, num_classes


def write_csv(filename, data, sep='\t'):
    with open(filename, 'w') as fp:
        a = csv.writer(fp, delimiter='\t')
        a.writerows(data)



