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

from scipy import stats, integrate
import seaborn as sns
sns.set(color_codes=True)

pandas2ri.activate()
numpy2ri.activate()
r = robjects.r
r_source = r['source']
r_source('../Evaluation/evaluate_model.R', **{'print.eval': True})


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

    if FLAGS.use_balanced:
        transp = True
    else:
        transp = False

    # Open DataFile

    start_time = time.time()
#     datafile, (mapped_labels, label_map) = load_data('TPM', label_col=9, transpose=True)
#     labelfile = load_data('Labels')
    datafile_orig, labels, (mapped_labels_df, label_map) = load_data('Linarsson', transpose=transp)
#     datafile, labels, (mapped_labels_df, label_map) = load_data(dataset='Allen', d_type='TPM', label_col=7, transpose=transp)

    mapped_labels = np.reshape(mapped_labels_df.values, (mapped_labels_df.shape[0],))
#     print(label_map)

    print("Data Loaded. Duration:", time.time() - start_time)
    np.set_printoptions(threshold=np.nan)

    if transp:
        a = Adasyn(datafile_orig, mapped_labels, label_map[:,1], beta=1)
        datafile, mapped_labels = a.balance_all()
    #     a.save_data(pjoin(FLAGS.data_dir, 'TPM_balanced_data.csv'), pjoin(FLAGS.data_dir, 'Mapped_Labels_inOrder_balanced.csv'))
        del(a)

    recr_labels = pd.DataFrame(data=mapped_labels).replace(label_map[:,1].tolist(), label_map[:,0].tolist())
#     print(recr_labels)

    # Data Normalization
    start_time = time.time()
    datafile_norm = normalize_data(datafile, transpose=transp)
    datafile_orig = normalize_data(datafile_orig, transpose=transp)

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

    # Run pretraining step
    sdae = SDAE.pretrain_sdae(input_x=data, shape=sdae_shape)
    del(data)

#     print("Random Forest Before Finetuning")
#     run_rf(datafile_norm, mapped_labels, sdae.get_weights, sdae.get_biases)

    # Create explanatory plots/graphs
    analyze(sdae, datafile_norm, recr_labels, mapped_labels, prefix='recr_Pretraining')
    analyze(sdae, datafile_orig, labels, mapped_labels, prefix='Pretraining')
        
#         pcafile = r.paste("Layer_{}".format(layer.which), "PCA.pdf", sep="_")
#         grdevices.pdf(pjoin(FLAGS.output_dir, pcafile))
#         r.par(mfrow=r.c(1,2))
#         p = r.prcomp(act)
#         # btype : broad_type
#         col = typeCols[btype[r.rownames(datafile)]]
#         r.plot(p.rx2('x'), col=col, pch=20)
#         r.plot(p.rx2('x')[:][2:3],col=col, pch=20)  
#         grdevices.dev_off()


    print("\nLoading train and test data-sets for Finetuning")
    data = load_data_sets(datafile_norm, mapped_labels)
    print("\nTotal Number of Examples:", data.train.num_examples + data.test.num_examples)

    # Run finetuning step    
    sdae = SDAE.finetune_sdae(sdae=sdae, input_x=data, n_classes=num_classes, label_map=label_map[:,0])

#     print("Random Forests After Finetuning for Autoencoder layers:")
#     run_rf(datafile_norm, mapped_labels, sdae.get_weights, sdae.get_biases, n_layers=nHLay)
#     print("Random Forests After Finetuning for all layers:")
#     run_rf(datafile_norm, mapped_labels, sdae.get_weights, sdae.get_biases)

    # Create explanatory plots/graphs
    analyze(sdae, datafile_norm, recr_labels, mapped_labels, prefix='recr_Finetuning')
    analyze(sdae, datafile_orig, labels, mapped_labels, prefix='Finetuning')

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


def analyze(sdae, datafile_norm, labels, mapped_labels, prefix):
    def_colors = robjects.globalenv['def_colors']
    do_analysis = robjects.globalenv['do_analysis']

#     labels.reset_index(level=0, inplace=True)
    def_colors(labels)
    act = np.float32(datafile_norm)

    for layer in sdae.get_layers:
        fixed = False if layer.which > sdae.nHLayers - 1 else True

        try:
            act = sdae.get_activation(act, layer.which, use_fixed=fixed)
            print("Analysis for layer {}:".format(layer.which + 1))
            temp = pd.DataFrame(data=act)
            do_analysis(temp, pjoin(FLAGS.output_dir, "{}_Layer_{}".format(prefix, layer.which)))
            
#             if not fixed:
#                 weights = sdae.get_weights[layer.which]
#                 for node in weights.transpose():
#                     sns.distplot(node, kde=False, fit=stats.gamma, rug=True);
#                     sns.plt.show()
                    
            plot_tSNE(act, mapped_labels, plot_name="Pyhton_{}_tSNE_layer_{}".format(prefix, layer.which))
        except FailedPreconditionError as e:
            break

if __name__ == '__main__':
    total_time = time.time()
    main()
    print("\n{}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    print("Total time:", time.time() - total_time)
    
