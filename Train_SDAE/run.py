import os
import shutil
import time
import stacked_dae as SDAE

from tools.config import FLAGS, home_out
from tools.start_tensorboard import start
from tools.data_handler import load_data

from tools.utils import load_data_sets_pretraining, load_data_sets
from tools.utils import normalize_data, label_metadata

_data_dir = FLAGS.data_dir
_output_dir = FLAGS.output_dir
_summary_dir = FLAGS.summary_dir
_chkpt_dir = FLAGS.chkpt_dir

def _check_and_clean_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.mkdir(d)


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
    
    """
    TODO: Next step to make it as much autonomous as possible.
        Load data and labels(if existed) here or in the run.py.
        Extract infromation from the data and the labels for the procedure to follow.
        If there are labels existed form the NN accordingly, while if not, either
        have a predefined configuration or create it on the fly by analyzing information
        from the input data.
    """
    
    # Open DataFile

    # Allan's Data
    start_time = time.time()
    datafile = load_data('RPKM', transpose=False)
    labelfile = load_data('Labels')
    print "Data Loaded. Duration:", time.time() - start_time

    # Data Normalization
    start_time = time.time()
    datafile_norm = normalize_data(datafile, transpose=False)
    print "Data Normalized. Duration:", time.time() - start_time

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
    print sdae_shape
    
    
    sdae = SDAE.pretrain_sdae(input_x=data, shape=sdae_shape)
    
    data = load_data_sets(datafile_norm, mapped_labels)
    sdae = SDAE.finetune_sdae(sdae=sdae, input_x=data, n_classes=num_classes)

if __name__ == '__main__':
    main()
