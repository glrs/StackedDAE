from __future__ import print_function

import os
import shutil
import time
import stacked_dae as SDAE

from os.path import join as pjoin
import numpy as np
import pandas as pd

from tools.config import FLAGS, home_out
from tools.start_tensorboard import start
from tools.data_handler import load_data, load_linarsson

from tools.utils import load_data_sets_pretraining, load_data_sets
from tools.utils import normalize_data, label_metadata, write_csv
from tools.ADASYN import Adasyn, all_indices
from tools.evaluate_model import run_random_forest as run_rf
from tools.evaluate_model import plot_tSNE

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from tensorflow.python.framework.errors import FailedPreconditionError

pandas2ri.activate()
numpy2ri.activate()
r = robjects.r

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
        Load data and labels(if existed).
        Extract information from the data and the labels for the procedure to follow.
        If there are labels existed form the NN accordingly, while if not, either
        have a predefined configuration or create it on the fly by analyzing information
        from the input data.
    """
    
    # Open DataFile

    # Allan's Data
    start_time = time.time()
    # datafile, (mapped_labels, label_map) = load_data('TPM', label_col=9, transpose=True)
    # labelfile = load_data('Labels')
    datafile, labels, (mapped_labels, label_map) = load_data('Linarsson', transpose=False)
#     datafile, labels, (mapped_labels, label_map) = load_data(dataset='Allen', d_type='TPM', label_col=9, transpose=False)

#     print(mapped_labels, label_map)
#     print(labels)

    print("Data Loaded. Duration:", time.time() - start_time)
    np.set_printoptions(threshold=np.nan)

#     write_csv(pjoin(FLAGS.data_dir, "Labels_1.csv"), mapped_labels.tolist())

#     a = Adasyn(datafile, mapped_labels, label_map[:,1], beta=0.5)
#     datafile, mapped_labels = a.balance_all()
#     a.save_data(pjoin(FLAGS.data_dir, 'TPM_balanced_data.csv'), pjoin(FLAGS.data_dir, 'Mapped_Labels_inOrder_balanced.csv'))
#     del(a)

    # Data Normalization
    start_time = time.time()
    datafile_norm = normalize_data(datafile, transpose=False)
#     datafile_norm = normalize_data(datafile, transpose=False)
    print("Data Normalized. Duration:", time.time() - start_time)

    # Get Label Metadata
#   mapped_labels, label_map = label_metadata(label_matrix=labelfile, label_col=7)
    num_classes = label_map.shape[0]

            
    print("\nLabel Counts:")
    for i in xrange(num_classes):
        print("{: >30}\t".format(label_map[i,0]), len(all_indices(i, mapped_labels.tolist())))

    # Get data-sets (train, test) in a proper way
    print("\nLoading train and test data-sets for pre-training")
    data = load_data_sets_pretraining(datafile_norm, split_only=False)
    print("\nTotal Number of Examples:", data.train.num_examples + data.test.num_examples)
    
    nHLay = FLAGS.num_hidden_layers
    nHUnits = [getattr(FLAGS, "hidden{0}_units".format(j + 1)) for j in xrange(nHLay)]
    
    # Get the number of existed features (e.g. genes) in the data-set 
    num_features = datafile_norm.shape[1]
    # Create the shape of the AutoEncoder
    sdae_shape = [num_features] + nHUnits + [num_classes]
    print(sdae_shape)
    
    
    sdae = SDAE.pretrain_sdae(input_x=data, shape=sdae_shape)
    del(data)
    
#     print("Random Forest Before Finetuning")
#     run_rf(datafile_norm, mapped_labels, sdae.get_weights, sdae.get_biases)

    r_source = r['source']
    r_source('../Evaluation/evaluate_model.R', **{'print.eval': True})

    



    analyze(sdae, datafile_norm, labels)

        
#         pcafile = r.paste("Layer_{}".format(layer.which), "PCA.pdf", sep="_")
#         grdevices.pdf(pjoin(FLAGS.output_dir, pcafile))
#         r.par(mfrow=r.c(1,2))
#         p = r.prcomp(act)
#         
#         # btype : broad_type
#         col = typeCols[btype[r.rownames(datafile)]]
#         r.plot(p.rx2('x'), col=col, pch=20)
#         r.plot(p.rx2('x')[:][2:3],col=col, pch=20)
#         
#         grdevices.dev_off()
        
#         plot_tSNE(act, mapped_labels, plot_name="tSNE_pre_layer_{}".format(layer.which))
    
        
    print("\nLoading train and test data-sets for Finetuning")
    data = load_data_sets(datafile_norm, mapped_labels)
    print("\nTotal Number of Examples:", data.train.num_examples + data.test.num_examples)
    
    sdae = SDAE.finetune_sdae(sdae=sdae, input_x=data, n_classes=num_classes, label_map=label_map[:,0]) #['broad_type']
#     print("Random Forests After Finetuning for Autoencoder layers:")
#     run_rf(datafile_norm, mapped_labels, sdae.get_weights, sdae.get_biases, n_layers=nHLay)
#     print("Random Forests After Finetuning for all layers:")
#     run_rf(datafile_norm, mapped_labels, sdae.get_weights, sdae.get_biases)

    analyze(sdae, datafile_norm, labels)

#     act = np.float32(datafile_norm)
#     for layer in sdae.get_layers:
#         fixed = False if layer.which > nHLay - 1 else True
#         print("t-SNE for layer ", layer.which + 1)
#         act = sdae.get_activation(act, layer.which, use_fixed=fixed)
#         plot_tSNE(act, mapped_labels, plot_name="tSNE_layer_{}".format(layer.which))

    
    print("\nConfiguration:")
    print("\n{: >45}\t".format("# Hidden Layers:"), nHLay)
    print("{: >45}\t".format("# Hidden Units:"), nHUnits)
    noise_ratios = [getattr(FLAGS, "noise_{0}".format(i)) for i in xrange(1,nHLay+1)]
    print("{: >45}\t".format("Noise Ratio (per layer):"), [row[0] for row in noise_ratios])
    print("{: >45}\t".format("Noise Type (MN, SP, TFDO):"), [row[1] for row in noise_ratios])
    if FLAGS.emphasis:
        print("{: >45}\t".format("Emphasis (Double, Full, No):"), FLAGS.emphasis_type)
    else:
        print("{: >45}\t".format("Emphasis (Double, Full, No):"), "No")
    l_rates = [getattr(FLAGS, "pre_layer{}_learning_rate".format(i)) for i in xrange(1,nHLay+1)]
    print("{: >45}\t".format("Unsupervised Learning Rate (per layer?):"), l_rates)
    print("{: >45}\t".format("Supervised Learning Rate:"), FLAGS.supervised_learning_rate)
    print("{: >45}\t".format("Batch size:"), FLAGS.batch_size)
    print("{: >45}\t".format("# Pretraining epochs:"), FLAGS.pretraining_epochs)
    print("{: >45}\t".format("# Finetuning epochs:"), FLAGS.finetuning_epochs)
#     Activation Function (Sigmoid, Tanh, ReLU)
#     Weight Initialization (Sigmoid, Tanh, ReLU)
#     Loss Function (X-Entropy, sum of sq. error)


def analyze(sdae, datafile_norm, labels):
    def_colors = robjects.globalenv['def_colors']
    labels.reset_index(level=0, inplace=True)
    def_colors(labels)
    do_analysis = robjects.globalenv['do_analysis']
    act = np.float32(datafile_norm)
    for layer in sdae.get_layers:
        fixed = False if layer.which > sdae.nHLayers - 1 else True
#         if layer.which > nHLay - 1:
#             break
        print("Analysis for layer {}:".format(layer.which + 1))
        try:
            act = sdae.get_activation(act, layer.which, use_fixed=fixed)
            temp = pd.DataFrame(data=act, index=labels[[0]]).reset_index(level=0)
            do_analysis(temp, pjoin(FLAGS.output_dir, "Layer_{}".format(layer.which)))
        except FailedPreconditionError as e:
            break

if __name__ == '__main__':
    total_time = time.time()
    main()
    print("\n{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    print("Total time:", time.time() - total_time)
    
