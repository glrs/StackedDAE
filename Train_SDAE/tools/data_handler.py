""" Data Handler for Allan's Data-set """

import pandas as pd
import os
import gzip
import numpy as np

from os.path import join as pjoin
from config import FLAGS

TPM = ['TPM_ready_data.csv', 'GSE71585_RefSeq_TPM.csv', 'GSE71585_RefSeq_TPM.csv.gz']
RPKM = ['RPKM_ready_data.csv', 'GSE71585_RefSeq_RPKM.csv', 'GSE71585_RefSeq_RPKM.csv.gz']
COUNTS = ['Counts_ready_data.csv', 'GSE71585_RefSeq_counts.csv', 'GSE71585_RefSeq_counts.csv.gz']
LABELS = ['Labels_inOrder.csv', 'GSE71585_Clustering_Results.csv', 'GSE71585_Clustering_Results.csv.gz']
#'counts_ordered_nonzero_zeroone.tsv', 
#'metadata_ordered_subset.tsv', 

def extract_data(in_f, out_f):
    print("Extracting", in_f)
    in_file = gzip.open(in_f, 'rb')
    out_file = open(out_f, 'wb')
    out_file.write(in_file.read())
    in_file.close()
    out_file.close()


def order_labels(data_in, label_in, data_out=None, label_out=None):
    print("Ordering Data with Labels...")
    
    labels = pd.read_csv(label_in, index_col=0)
    data = pd.read_csv(data_in, index_col=0)
    
    common_labels = labels.index.intersection(data.columns)
#     common_labels2 = data.columns.intersection(labels.index)
    
#     data_nonzero = data.loc[(data > 0).any(axis=1)].dropna()
    data_nonzero = data[(data.sum(axis=1) > 0)].dropna()
    data_nonzero = data_nonzero[common_labels]
    
    """ Better here with non_zero than above? """
    common_labels2 = data_nonzero.columns.intersection(labels.index)
    label_sub = labels.loc[common_labels2]
    label_sub.index.names = labels.index.names
    
    label_sub_sort = label_sub.sort_index(0)
    data_sub_sort = data_nonzero.reindex_axis(sorted(data_nonzero.columns), axis=1)
    
    # Check that it worked
    assert(data_sub_sort.columns == label_sub_sort.index).all()
    
    if data_out is not None and label_out is not None:
        data_sub_sort.to_csv(data_out, sep="\t")
        label_sub_sort.to_csv(label_out, sep="\t")

    return data_sub_sort, label_sub_sort


def sort_labels(data_in):
    d = pd.read_csv(data_in, sep='\t', index_col=0)
    return d.sort_index(0)


def load_data(d_type=None, transpose=False):
    if d_type == 'TPM':
        data = TPM
        print("TPM file is loading...")
    elif d_type == 'RPKM':
        data = RPKM
        print("RPKM file is loading...")
    elif d_type == 'Counts':
        data = COUNTS
        print("Counts file is loading...")
    elif d_type == 'Labels':
        data = LABELS
        print("Label file is loading...")
    else:
        exit("Usage: load_data(data_type=['TPM', 'RPKM', 'Counts', 'Labels'])")
        
    if not os.path.exists(pjoin(FLAGS.data_dir, data[0])):
        if not os.path.exists(pjoin(FLAGS.data_dir, data[1])):
            if not os.path.exists(pjoin(FLAGS.data_dir, data[2])):
                exit("You should download and place the data in the correct folder.")
            else:
                extract_data(pjoin(FLAGS.data_dir, data[2]), pjoin(FLAGS.data_dir, data[1]))
                if d_type == 'Labels':
                    exit("Labels extracted. You need to give a dataset first to receive the labels.")
                else:
                    if not os.path.exists(pjoin(FLAGS.data_dir, LABELS[1])):
                        extract_data(pjoin(FLAGS.data_dir, LABELS[2]), pjoin(FLAGS.data_dir, LABELS[1]))

                    d, _ = order_labels(pjoin(FLAGS.data_dir, data[1]), pjoin(FLAGS.data_dir, LABELS[1]),
                                        pjoin(FLAGS.data_dir, data[0]), pjoin(FLAGS.data_dir, LABELS[0]))
        else:
            if d_type == 'Labels':
                exit("You need to give a dataset first to receive the labels.")
            else:
                d, _ = order_labels(pjoin(FLAGS.data_dir, data[1]), pjoin(FLAGS.data_dir, LABELS[1]),
                                    pjoin(FLAGS.data_dir, data[0]), pjoin(FLAGS.data_dir, LABELS[0]))
    else:
        d = pd.read_csv(pjoin(FLAGS.data_dir, data[0]), sep='\t', index_col=0)
        

    if d_type == 'Labels':
        return d
    else:
        if transpose:
            return np.array(d.transpose())
        else:
            return np.array(d)


        