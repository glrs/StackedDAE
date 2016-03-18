#import matplotlib as mpl
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
from config import FLAGS
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interp

methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(pjoin(FLAGS.output_dir, title.replace(' ', '_') + '_CM.png'))
    plt.close()
    

def plot_roc_curve(y_pred, y_true, n_classes, title='ROC_Curve'):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    tresholds = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], tresholds[i] = roc_curve(y_true, y_pred, pos_label=i, drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), pos_label=0, drop_intermediate=True)
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    print("Thresholds:")
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        print("Class_{0}: {1}".format(i, tresholds[i]))

    # Average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    
    # Plot all ROC curves
    fig = plt.figure()
    ax = fig.add_subplot(111)

#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              linewidth=3, ls='--', color='red')
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=3, ls='--', color='green')
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic')
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    plt.savefig(pjoin(FLAGS.output_dir, title.replace(' ', '_') + '_ROC.png'), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
    

def hist_comparison(data1, data2):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle('Histogram Before and After Normalization')
    ax1.hist(data1, 10, facecolor='green', alpha=0.75)
    ax1.set_xlabel("Values")
    ax1.set_ylabel("# of Examples")
    ax1.grid(True)
    ax2.hist(data2, 10, facecolor='green', alpha=0.75)
    ax2.set_xlabel("Values")
    ax2.grid(True)

    f.savefig(pjoin(FLAGS.output_dir, 'hist_comparison.png'))
#     plt.show()
    plt.close()
    

def make_heatmap(data, name):
    f = plt.figure()
    ax1 = f.add_axes([0.1,0.1,0.8,0.8])
    ax1.imshow(data, interpolation="none")
    f.savefig(pjoin(FLAGS.output_dir, name + '.png'))
    plt.close()

def make_2d_hist(data, name):
    f = plt.figure()
    X,Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
    im = plt.pcolormesh(X,Y,data.transpose(), cmap='seismic')
    plt.colorbar(im, orientation='vertical')
#     plt.hexbin(data,data)
#     plt.show()
    f.savefig(pjoin(FLAGS.output_dir, name + '.png'))
    plt.close()
    
# def make_2d_hexbin(data, name):
#     f = plt.figure()
#     X,Y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
#     plt.hexbin(X, data)
# #     plt.show()
#     f.savefig(pjoin(FLAGS.output_dir, name + '.png'))

def heatmap_comparison(data1, label1, data2, label2, data3, label3):    
    interpolation = methods[1]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True) 
    fig.suptitle('Heatmap Comparison of Normal and Noisy Data')
    ax1.imshow(data3, interpolation=interpolation)
    ax1.set_title(label1)
    ax1.set_ylabel("Examples")
    ax1.set_xlabel("Features")
    ax1.set_aspect('equal')
    
    ax2.imshow(data2, interpolation=interpolation)
    ax2.set_title(label2)
    ax2.set_xlabel("Features")
    ax2.set_aspect('equal')
    
    ax3.imshow(data1, interpolation=interpolation)
    ax3.set_title(label3)
    ax3.set_xlabel("Features")
    ax3.set_aspect('equal')
    
    cax = fig.add_axes([0, 0, .1, .1])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0.5)
    cax.set_frame_on(True)
#     plt.colorbar(ax1, ax2, orientation='vertical')
    plt.show()
    plt.close()
#     
#     fig = plt.figure(figsize=(6, 3.2))
# 
#     ax = fig.add_subplot(111)
#     ax.set_title('colorMap')
#     plt.imshow(data1)
#     ax.set_aspect('equal')
#     
#     cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
#     cax.get_xaxis().set_visible(False)
#     cax.get_yaxis().set_visible(False)
#     cax.patch.set_alpha(0)
#     cax.set_frame_on(False)
#     plt.colorbar(orientation='vertical')
#     plt.show()
#     