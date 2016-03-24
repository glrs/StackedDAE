from __future__ import print_function

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
        os.makedirs(_output_dir)
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
    datafile = load_data('TPM', transpose=False)
    labelfile = load_data('Labels')
    print("Data Loaded. Duration:", time.time() - start_time)

    # Data Normalization
    start_time = time.time()
    datafile_norm = normalize_data(datafile, transpose=False)
    print("Data Normalized. Duration:", time.time() - start_time)

#     # Get data-sets (train, test) in a proper way
#     data = load_data_sets_pretraining(datafile_norm, split_only=False)
#     print("Num_Examples:", data.train.num_examples + data.test.num_examples)

    # Get Label Metadata
    mapped_labels, label_map = label_metadata(label_matrix=labelfile, label_col=7)
    num_classes = label_map.shape[0]

    print(label_map[:,0])
    
#     import numpy as np
#     import matplotlib.pyplot as pl
#     from tools.ADASYN import all_indices, all_indices_multi, getClassCount, getd, getG, getRis, generateSamples, join_with_the_rest
#     
#     idx = all_indices_multi([0,1,3,4,5,6,7], mapped_labels.tolist())
#     print("All indices ADASYN:", len(idx))
#     
#     print(idx)
#     # Get the minority and majority count
#     ms,ml = getClassCount(datafile_norm, mapped_labels.tolist(), label_map[:,1].tolist(), 2)
#     print("Ms - Ml :", ms, ml)
#     d = getd(datafile_norm, mapped_labels.tolist(), ms, ml)
#     G = getG(datafile_norm, mapped_labels.tolist(), ms, ml, beta=1)
#     print("d - G :", d, G)
#     
#     # Get the list of r values, which indicate how many samples will be made per data point in the minority dataset
#     rlist = getRis(datafile_norm, mapped_labels.tolist(), label_map[:,1].tolist(), 2, 5)
#     print("Rlist length :", len(rlist))
#     
#     # Generate the synthetic data
#     newX, newy = generateSamples(rlist,datafile_norm, mapped_labels.tolist(), G, 2, 5)
#     print("Counts:")
#     for i in xrange(8):
#         try:
#             print(label_map[i,0], "\t", len(all_indices(i, newy)))
#         except:
#             continue
# 
#     datafile_norm, mapped_labels = join_with_the_rest(datafile_norm, mapped_labels.tolist(), newX, newy, label_map[:,1].tolist(), 2)
    
    from tools.ADASYN import Adasyn, all_indices
    
    a = Adasyn(datafile_norm, mapped_labels, label_map[:,1])
    
    datafile_norm, mapped_labels = a.balance_all()
    del(a)
    print("Counts:")
    for i in xrange(8):
        print(label_map[i,0], "\t", len(all_indices(i,mapped_labels.tolist())))
    
    # Get data-sets (train, test) in a proper way
    data = load_data_sets_pretraining(datafile_norm, split_only=False)
    print("Num_Examples:", data.train.num_examples + data.test.num_examples)
    
    nHLay = FLAGS.num_hidden_layers
    nHUnits = [getattr(FLAGS, "hidden{0}_units".format(j + 1)) for j in xrange(nHLay)]
    
    # Get the number of existed features (e.g. genes) in the data-set 
    num_features = datafile_norm.shape[1]
    # Create the shape of the AutoEncoder
    sdae_shape = [num_features] + nHUnits + [num_classes]
    print(sdae_shape)
    
    
    sdae = SDAE.pretrain_sdae(input_x=data, shape=sdae_shape)
    del(data)
    data = load_data_sets(datafile_norm, mapped_labels)
    print("Num_Examples:", data.train.num_examples + data.test.num_examples)
    
    sdae = SDAE.finetune_sdae(sdae=sdae, input_x=data, n_classes=num_classes, label_map=label_map[:,0]) #['broad_type']

if __name__ == '__main__':
    total_time = time.time()
    main()
    print("Total time:", time.time() - total_time)
