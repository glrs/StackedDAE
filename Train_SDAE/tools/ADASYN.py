from sklearn.neighbors import NearestNeighbors
from random import choice

'''
Created on 14-jun.-2013
@author: Olivier.Janssens
'''

'''
Modified on 24-March-2016
@author: Anastasios Glaros
'''

import numpy as np
import random

class Adasyn(object):
    def __init__(self, data, labels, classes, K=5, beta=1):
        self.X = data
        self.K = K
        self.beta = beta
        self.new_X, self.new_y = [], []
        self.d, self.G = [], []
        
        try:
            assert not isinstance(classes, list)
            self.classes = classes.tolist()
        except AssertionError as e:
            self.classs = classes

        try:
            assert not isinstance(labels, list)
            self.y = labels.tolist()
        except AssertionError as e:
            self.y = labels
            
        temp = []
        for i in xrange(len(self.classes)):
            temp.append(len(all_indices(i, self.y)))
        
        self.majority_class = self.classes[temp.index(max(temp))] #np.where(np.asarray(temp)==max(temp))[0][0]]
                
        
    def balance_all(self):
        classes = np.copy(self.classes).tolist()
        classes.remove(self.majority_class)

        print "Classes:", classes

        # Loop for all the classes except the majority
        for class_i in classes:
            print "\nFor class: ", class_i
            ms, ml = self.get_class_count(self.X, self.y, class_i, self.majority_class)

            d = self.get_d(self.X, self.y, ms, ml)
            G = self.get_G(self.X, self.y, ms, ml, self.beta)
            
            rlist = self.get_Ris(self.X, self.y, class_i, self.K)
#             print("ms, ml, d, G, len(rlist): ", ms, ml, d, G, len(rlist))
            
            new_X, new_y = self.generate_samples(rlist, self.X, self.y, G, class_i, self.K)
            print "Length of original_X, new_X:", ms, len(new_X)
#             print("shape of new_X, new_y:", new_X.shape, new_y.shape)
            self.new_X.append(new_X)
            self.new_y.append(new_y)
        
        return self.join_all_together()
#             X, y = self.join_with_the_rest(self.X, self.y, newX, newy, self.classes, class_i)

    def save_data(self, data_filename, label_filename):
        from tools.utils import write_csv
        import csv
        print(type(self.new_X), "saving...")
        with open(data_filename, "wb") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(self.new_X)
            
        print("Saved.")
#         write_csv(data_filename, self.new_X)
        del(self.new_X)

#         with open(label_filename, "wb") as f:
#             writer = csv.writer(f, delimiter='\t')
#             writer.writerows(self.new_y)
        write_csv(label_filename, self.new_y)
        del(self.new_y)

    # @param: X The datapoints e.g.: [f1, f2, ... ,fn]
    # @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
    # @return ms: The amount of samples in the minority group
    # @return ms: The amount of samples in the majority group
    def get_class_count(self, X, y, minorityclass, majorityclass):
        indicesZero = all_indices(minorityclass, y)
        indicesOne = all_indices(majorityclass, y)
        
        if len(indicesZero) > len(indicesOne):
            ms = len(indicesOne)
            ml = len(indicesZero)
        else:
            ms = len(indicesZero)
            ml = len(indicesOne)
        return ms,ml

    # @param: X The datapoints e.g.: [f1, f2, ... ,fn]
    # @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
    # @param ms: The amount of samples in the minority group
    # @param ms: The amount of samples in the majority group
    # @return: The ratio between the minority and majority group
    def get_d(self, X,y,ms,ml):
    
        return float(ms)/float(ml)

    # @param: X The datapoints e.g.: [f1, f2, ... ,fn]
    # @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
    # @param ms: The amount of samples in the minority group
    # @param ms: The amount of samples in the majority group
    # @return: the G value, which indicates how many samples should be generated in total, this can be tuned with beta
    def get_G(self, X,y,ms,ml,beta):
        return (ml-ms)*beta


    # @param: X The datapoints e.g.: [f1, f2, ... ,fn]
    # @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
    # @param: minorityclass: The minority class
    # @param: K: The amount of neighbours for Knn
    # @return: rlist: List of r values
    def get_Ris(self, X,y, minorityclass=0, K=5):
        indicesMinority = all_indices(minorityclass,y)
        ymin = np.array(y)[indicesMinority]
        Xmin = np.array(X)[indicesMinority]
        neigh = NearestNeighbors(n_neighbors=30,algorithm = 'ball_tree')
        neigh.fit(X)
        
#         print "Shapes:", Xmin[0].shape, Xmin[0].reshape(1,-1).shape
        
        rlist = [0]*len(ymin)
        normalizedrlist = [0]*len(ymin)

        classes = np.copy(self.classes).tolist()
        classes.remove(minorityclass)
        
        for i in xrange(len(ymin)):
            indices = neigh.kneighbors(Xmin[i].reshape(1,-1), K, False)
            
            #print ">", len(all_indices_multi(classes, np.array(y)[indices].tolist()[0]))
            rlist[i] = float(len(all_indices_multi(classes, np.array(y)[indices].tolist()[0]))) / K
    
            
        normConst = sum(rlist)
    
        try:
            for j in xrange(len(rlist)):
                normalizedrlist[j] = (rlist[j]/normConst)
        except ZeroDivisionError as e:
            normalizedrlist = rlist
            print(rlist)
    
        return normalizedrlist
        

    # @param: rlist: List of r values
    # @param: X The datapoints e.g.: [f1, f2, ... ,fn]
    # @param: y the classlabels e.g: [0,1,1,1,0,...,Cn]
    # @return: the G value, which indicates how many samples should be generated in total, this can be tuned with beta
    # @param: minorityclass: The minority class
    # @param: K: The amount of neighbours for Knn
    # @return: The synthetic data samples
    def generate_samples(self, rlist,X,y,G,minorityclasslabel,K):
        syntheticdata = []
        
        indicesMinority = all_indices(minorityclasslabel,y)
        ymin = np.array(y)[indicesMinority]
        Xmin = np.array(X)[indicesMinority]
        
#         print "Xmin shape: ", Xmin.shape, ", len of ymin:", len(ymin)
        
        neigh = NearestNeighbors(n_neighbors=30,algorithm = 'ball_tree')
        neigh.fit(Xmin)
        gsum=0
        for k in xrange(len(ymin)):
            g = int(np.round(rlist[k]*G))
            #print g, "= int round ", rlist[k], "*", G
            gsum += g
            for l in xrange(g):
                ind = random.choice(neigh.kneighbors(Xmin[k].reshape(1,-1),K,False)[0])
                s = Xmin[k] + (Xmin[ind]-Xmin[k]) * random.random()
                syntheticdata.append(s)
        
#         print "synthetic shape: ", np.asarray(syntheticdata).shape, ", gsum:", gsum
        
        try:
            new_data = np.concatenate((syntheticdata, Xmin),axis=0)
            new_y = [minorityclasslabel] * len(new_data)
        except ValueError as e:
            new_data = Xmin
            new_y = ymin

        return new_data, new_y

    def join_all_together(self):
        X_all, y_all = [], []
        classes = np.copy(self.classes).tolist()
        classes.remove(self.majority_class)
        print "\nJoining Original and Synthetic datasets..."
        # Loop for all classes except 1 (the majority class)
        for i, class_i in zip(xrange(len(self.classes) - 1), classes):
            classes_no_minor = np.copy(self.classes).tolist()
            classes_no_minor.remove(class_i)
#             print i, class_i, classes_no_minor

            if i == 0:
                indicesMajority = all_indices_multi(classes_no_minor, self.y)
                ymaj = np.array(self.y)[indicesMajority]
                Xmaj = np.array(self.X)[indicesMajority]
#                 print "Indices_Majority:", len(indicesMajority), "len ymaj:", len(ymaj), "len Xmaj:", len(Xmaj), "len self.new_X:", len(self.new_X)

#                 X_all = np.concatenate((Xmaj, self.new_X[i]), axis=0)
#                 y_all = np.concatenate((ymaj, self.new_y[i]), axis=0)
            else:
                indicesMajority = all_indices_multi(classes_no_minor, y_all.tolist())
                ymaj = y_all[indicesMajority]
                Xmaj = X_all[indicesMajority]
#                 print "Indices_Majority:", len(indicesMajority), "len ymaj:", len(ymaj), "len Xmaj:", len(Xmaj), "len self.new_X:", len(self.new_X)

#                 X_all = np.concatenate((X_all, np.concatenate((Xmaj, self.new_X[i]), axis=0)), axis=0)
#                 y_all = np.concatenate((y_all, np.concatenate((ymaj, self.new_y[i]), axis=0)), axis=0)
            
            X_all = np.concatenate((Xmaj, self.new_X[i]), axis=0)
            y_all = np.concatenate((ymaj, self.new_y[i]), axis=0)
        print "Joined. Length of X_all and y_all:", len(X_all), len(y_all)

        return X_all, y_all
    
    def join_with_the_rest(self, X,y,newData,newy,classes, minorityclass):
        classes.remove(minorityclass)
        indicesMajority = all_indices_multi(classes, y)
        ymaj = np.array(y)[indicesMajority]
        Xmaj = np.array(X)[indicesMajority]
    
        return np.concatenate((Xmaj,newData),axis=0), np.concatenate((ymaj,newy),axis=0)
    
    def joinwithmajorityClass(self, X,y,newData,newy,majorityclasslabel):
        indicesMajority = all_indices(majorityclasslabel,y)
        ymaj = np.array(y)[indicesMajority]
        Xmaj = np.array(X)[indicesMajority]
    
        return np.concatenate((Xmaj,newData),axis=0),np.concatenate((ymaj,newy),axis=0)

# @param value: The classlabel
# @param qlist: The list in which to search
# @return: the indices of the values that are equal to the classlabel
def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices


# @param values: The classlabels except the minority's class
# @param qlist: The list in which to search
# @return: the indices of the values that are equal to the classlabel
def all_indices_multi(values, qlist):
    indices = []
    for i in xrange(len(values)):
        idx = -1
        flag = True
        while flag:            
            try:
                idx = qlist.index(values[i], idx+1)
                indices.append(idx)
            except ValueError:
                flag = False
    return indices

