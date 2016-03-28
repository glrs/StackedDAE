import numpy as np
from scipy.special import expit
from sklearn import ensemble

def get_activations(exp_data, w, b):
    exp_data = np.transpose(exp_data)
    prod = exp_data.dot(w)
    prod_with_bias = prod + b
    return( expit(prod_with_bias) )

# Order of *args: first all the weights and then all the biases
def run_random_forest(exp_data, labels, weights, biases, n_layers=None):
#     print len(exp_data), len(labels)
    assert len(exp_data) == len(labels)
    
    # I think they should be already transposed when running the code. Will see
    act = exp_data#.T
    
    # Using ternary operator for shortness
    n = n_layers if n_layers else len(weights)
    
    for i in range(n):
        print('Weights and biases for layer: ' + str(i+1))
#         print np.asarray(weights[i]).shape, np.asarray(biases[i]).shape
        act = get_activations(act.T, weights[i], biases[i])
        
    rf = ensemble.RandomForestClassifier(n_estimators=1000, oob_score=True, max_depth=5)
    rfit = rf.fit(act, labels)
    print('OOB score: %.2f\n' % rfit.oob_score_)

