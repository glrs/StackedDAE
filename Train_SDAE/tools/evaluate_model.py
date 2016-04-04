import numpy as np
import sklearn
from scipy.special import expit
from sklearn import ensemble
from sklearn.manifold import TSNE
import time
from os.path import join as pjoin
from tools.config import FLAGS
from tools.visualize import scatter

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


def plot_tSNE(data, labels, random_state=7074568, plot_name='tsne-generated_{}.png'):
    # Calculate t-SNE projections
    x_projection = TSNE(random_state=random_state).fit_transform(data)
    
    # Form the output file name
    plot_name = plot_name if plot_name.find(".") > 0 else plot_name+".png"
    plot_name = pjoin(FLAGS.output_dir, plot_name.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    
    # Create and save a t-SNE scatter plot
    scatter(x_projection, labels, plot_name=plot_name)

