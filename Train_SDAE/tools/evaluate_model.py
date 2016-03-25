import numpy as np
# import pandas as pd
# import sys
from scipy.special import expit
from sklearn import ensemble

def get_activations(exp_data, w, b):
    exp_data = np.transpose(exp_data)
    prod = exp_data.dot(w)
    prod_with_bias = prod + b
    return( expit(prod_with_bias) )

# Order of *args: first all the weights and then all the biases
def run_random_forest(nHLayers, exp_data, labels, *args):
    print len(args[0]), len(args[0][0]), len(args[0][1])
    print len(args[0][2])
    print "NewLine!\n", len(args[0][3])
    print "NewLine!\n", len(args[0][4])
    assert len(exp_data) == len(labels)
    
    # I think they should be already transposed when running the code. Will see
    act = exp_data#.T
    
    for i in range(nHLayers):
        print('Weights and biases for layer: ' + str(i+1))
        print np.asarray(args[0][i]).shape, np.asarray(args[0][nHLayers + i]).shape
        act = get_activations(act.T, args[0][i], args[0][nHLayers + i])
        
    rf = ensemble.RandomForestClassifier(n_estimators=1000, oob_score=True, max_depth=5)
    rfit = rf.fit(act, labels)
    print('OOB score: %.2f\n' % rfit.oob_score_)

