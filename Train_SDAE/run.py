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
from tools.data_handler import load_data, load_linarsson_labels, load_extra

from tools.utils import load_data_sets_pretraining, load_data_sets
from tools.utils import normalize_data, label_metadata, write_csv
from tools.ADASYN import Adasyn, all_indices
from tools.evaluate_model import run_random_forest as run_rf
from tools.evaluate_model import plot_tSNE
from tools.evaluate import predict

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from tensorflow.python.framework.errors import FailedPreconditionError

from scipy import stats, integrate
import seaborn as sns
from rpy2.rinterface._rinterface import RRuntimeError
sns.set(color_codes=True)

pandas2ri.activate()
numpy2ri.activate()
r = robjects.r
r_source = r['source']
r_source('../Evaluation/evaluate_model.R', **{'print.eval': True})

np.set_printoptions(threshold=np.nan)

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

    # Set Hyper-parameters
    nHLay = FLAGS.num_hidden_layers
    nHUnits = [getattr(FLAGS, "hidden{0}_units".format(j + 1))\
               for j in xrange(nHLay)]
    
    if FLAGS.use_balanced:
        transp = True
    else:
        transp = False

    # Read/Upload/Process the Data
    start_time = time.time()
#     datafile, (mapped_labels, label_map) = load_data('TPM', label_col=9, transpose=True)
    datafile, labels, meta = load_data(FLAGS.dataset, d_type='filtered',\
                                       label_col=1, transpose=transp)

#     datafile_orig, labels, meta = load_data(FLAGS.dataset, d_type='filtered',\
#                                             label_col=7, transpose=transp)
    
    print("Data Loaded. Duration:", time.time() - start_time)
    
    
    # Receive/Set Metadata (Labels)
    mapped_labels_df, label_map = meta
    mapped_labels = np.reshape(mapped_labels_df.values, (mapped_labels_df.shape[0],))
    num_classes = label_map.shape[0]
    print("\nClass Statistics:")
    for i in xrange(num_classes):
        print("{: >30}\t".format(label_map[i,0]),\
              len(all_indices(i, mapped_labels.tolist())))

    
    # Balance classes
    balanced_data = None
    recr_labels = None
    if transp:
        a = Adasyn(datafile, mapped_labels, label_map[:,1], beta=1)
        balanced_data, mapped_labels = a.balance_all()
#         a.save_data(pjoin(FLAGS.data_dir, 'TPM_balanced_data.csv'),\
#                     pjoin(FLAGS.data_dir, 'Mapped_Labels_inOrder_balanced.csv'))

        recr_labels = pd.DataFrame(data=mapped_labels)
        recr_labels = recr_labels.replace(label_map[:,1].tolist(),\
                                          label_map[:,0].tolist())

    data = balanced_data if transp else datafile

    # Data Normalization
    start_time = time.time()
    norm_data = normalize_data(data, transpose=transp)
    
    norm_orig = None
    if transp:
        norm_orig = normalize_data(datafile, transpose=transp)

    print("Data Normalized. Duration:", time.time() - start_time)


    # Get the number of existed features (e.g. genes) in the data-set 
    num_features = norm_data.shape[1]

    # Create the shape of the AutoEncoder
    sdae_shape = [num_features] + nHUnits + [num_classes]
    print(sdae_shape)


###  ---  Pre-training Phase  ---  ###

    # Get data-sets (train, test) for pretraining in a proper way
    data = load_data_sets_pretraining(norm_data, split_only=False)
#     print("\nTotal Number of Examples:",\
#           data.train.num_examples + data.test.num_examples)


    # Run pretraining step
    sdae = SDAE.pretrain_sdae(input_x=data, shape=sdae_shape)
    del(data)

    
    # Run Random Forest Before Finetuning ### np.insert(norm_data, 1, np.ones_like((norm_data[:,0])), 1)
    run_rf(norm_data, mapped_labels, sdae.get_weights,\
           sdae.get_biases, bias_node=True)


    # Load another dataset to test it on the created model
#     sub_labels, _ = load_linarsson_labels(sub_labels=True)
    data_an, labels_an, meta = load_extra('Allen',\
                                          'TPM_common_ready_data.csv',\
                                          transpose=True, label_col=7)
    
    data_an = normalize_data(data_an, transpose=False)
    data_an = np.transpose(data_an)
#     data_an = np.insert(data_an, 1, np.ones_like((data_an[:,0])), 1)
    
    mapped_an_df, l_map = meta
    mapped_an_labs = np.reshape(mapped_an_df.values, (mapped_an_df.shape[0],))

    # Create comprehensive plots/graphs
    try:
        analyze(sdae, data_an, labels_an, bias_node=True, prefix='Foreign_Pretraining')
        analyze(sdae, np.insert(norm_orig, 1, np.ones_like((norm_orig[:,0])), 1),\
                labels, prefix='Pretraining')
    except:
        pass
#     analyze(sdae, datafile_norm, recr_labels, prefix='recr_Pretraining')
#     analyze(sdae, datafile_norm, sub_labels, mapped_labels, prefix='recr_Pretraining')
    

###  ---  Fine-tuning Phase  ---  ###

    # Get data-sets (train, test) for finetuning in a proper way
    data = load_data_sets(norm_data, mapped_labels)
#     print("\nTotal Number of Examples:",\
#           data.train.num_examples + data.test.num_examples)


    # Run finetuning step
    sdae = SDAE.finetune_sdae(sdae=sdae, input_x=data,\
                              n_classes=num_classes,\
                              label_map=label_map[:,0])


    foreign_data = load_data_sets(data_an, mapped_an_labs, split_only=False)
    p, t = predict(sdae, foreign_data.all, bias_node=True)
    p = pd.DataFrame(data=p).replace(l_map[:,1].tolist(), l_map[:,0].tolist())
    t = pd.DataFrame(data=t).replace(l_map[:,1].tolist(), l_map[:,0].tolist())
    print(p, t)
    p.to_csv(pjoin(FLAGS.output_dir, 'Predictions_of_Foreign.txt'), sep='\t')
    t.to_csv(pjoin(FLAGS.output_dir, 'True_labels_of_Foreign.txt'), sep='\t')


    # Run Random Forests After Finetuning for all layers
    run_rf(norm_data, mapped_labels, sdae.get_weights,\
           sdae.get_biases, bias_node=True)
#     run_rf(datafile_norm, mapped_labels, sdae.get_weights, sdae.get_biases, n_layers=nHLay)
#     print("Random Forests After Finetuning for all layers:")


    # Create comprehensive plots/graphs
#     analyze(sdae, datafile_norm, recr_labels, mapped_labels, prefix='recr_Finetuning')
    try:
        analyze(sdae, data_an, labels_an, mapped_labels, bias_node=True,\
                prefix='Foreign_Finetuning')
        analyze(sdae, norm_orig, labels, mapped_labels, bias_node=True,\
                prefix='Finetuning')
    except:
        pass

    # print the used set up
    print_setup()


def analyze(sdae, datafile_norm, labels, mapped_labels=None, bias_node=False, prefix=None):
    def_colors = robjects.globalenv['def_colors']
    do_analysis = robjects.globalenv['do_analysis']

#     labels.reset_index(level=0, inplace=True)
    def_colors(labels)
    act = np.float32(datafile_norm)

    try:
        do_analysis(act, sdae.get_weights, sdae.get_biases,\
                    pjoin(FLAGS.output_dir, "{}_R_Layer_".format(prefix)),\
                    bias_node=bias_node)
    except RRuntimeError as e:
        pass

#     for layer in sdae.get_layers:
#         fixed = False if layer.which > sdae.nHLayers - 1 else True
#  
#         try:
#             act = sdae.get_activation(act, layer.which, use_fixed=fixed)
#             print("Analysis for layer {}:".format(layer.which + 1))
#             temp = pd.DataFrame(data=act)
#             do_analysis(temp, pjoin(FLAGS.output_dir, "{}_Layer_{}".format(prefix, layer.which)))
#              
# #             if not fixed:
# #                 weights = sdae.get_weights[layer.which]
# #                 for node in weights.transpose():
# #                     sns.distplot(node, kde=False, fit=stats.gamma, rug=True);
# #                     sns.plt.show()
#             try:
#                 plot_tSNE(act, mapped_labels, plot_name="Pyhton_{}_tSNE_layer_{}".format(prefix, layer.which))
#             except IndexError as e:
#                 pass
#         except FailedPreconditionError as e:
#             break

def print_setup():
    nHLay = FLAGS.num_hidden_layers
    nHUnits = [getattr(FLAGS, "hidden{0}_units".format(j + 1)) for j in xrange(nHLay)]
    l_rates = [getattr(FLAGS, "pre_layer{}_learning_rate".format(i)) for i in xrange(1,nHLay+1)]
    noise_ratios = [getattr(FLAGS, "noise_{0}".format(i)) for i in xrange(1,nHLay+1)]
    print("\nConfiguration:")
    print("\n{: >45}\t".format("Dataset:"), FLAGS.dataset)
    print("\n{: >45}\t".format("# Hidden Layers:"), nHLay)
    print("{: >45}\t".format("# Hidden Units:"), nHUnits)
    print("{: >45}\t".format("Noise Ratio (per layer):"), [row[0] for row in noise_ratios])
    print("{: >45}\t".format("Noise Type (MN, SP, TFDO):"), [row[1] for row in noise_ratios])
    if FLAGS.emphasis:
        print("{: >45}\t".format("Emphasis (Double, Full, No):"), FLAGS.emphasis_type)
    else:
        print("{: >45}\t".format("Emphasis (Double, Full, No):"), "No")
    print("{: >45}\t".format("Unsupervised Learning Rate (per layer?):"), l_rates)
    print("{: >45}\t".format("Supervised Learning Rate:"), FLAGS.supervised_learning_rate)
    print("{: >45}\t".format("Batch size:"), FLAGS.batch_size)
    print("{: >45}\t".format("# Pretraining epochs:"), FLAGS.pretraining_epochs)
    print("{: >45}\t".format("# Finetuning epochs:"), FLAGS.finetuning_epochs)
#     Activation Function (Sigmoid, Tanh, ReLU)
#     Weight Initialization (Sigmoid, Tanh, ReLU)
#     Loss Function (X-Entropy, sum of sq. error)

if __name__ == '__main__':
    total_time = time.time()
    main()
    print("\n{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    print("Total time:", time.time() - total_time)
    
