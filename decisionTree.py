import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import math

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class Node():
    def __init__(self, level, val=None, col=None, childs=[]):
        self.value = val
        self.feature = col
        self.children = childs
        self.level = level


class DTClassifier(BaseEstimator,ClassifierMixin):
    """ Initialize class with chosen hyperparameters.
    Args:
        hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
        lr (float): A learning rate / step size.
        shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
    Example:
        DT  = DTClassifier()
    """
    def __init__(self, count, labels, shuffle=False):
        self.featureCounts = count
        self.featureLabels = labels
        self.shuffle = shuffle
        self.head = Node(0, -1)


    """ Fit the data; Make the Desicion tree
    Args:
        X (array-like): A 2D numpy array with the training data, excluding targets
        y (array-like): A 2D numpy array with the training targets
    Returns:
        self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
    """
    def fit(self, X, y):
        self.splitData(X, y, self.head, self.featureLabels.copy())
        print("*********PRINTING NODES***********")
        self.printTree(self.head)

        return self

    def getGains(self, array, labels):
        allData = self.concatLabels(array.reshape(-1,1), labels.reshape(-1,1))
        arySize = np.size(labels)
        vals = np.unique(allData[:,0])
        features = []
        sum = 0
        #split up by value (Y/N etc)
        for i in range(np.size(vals)):
            features.append(allData[allData[:,0] == vals[i]])
        #split each of those by output
        for feature in features: #i in range(np.size(features)):
            outps = np.unique(feature[:,1])
            p = np.size(feature,0)/arySize #outerCoef
            #sum output classes
            for outp in outps:
                hasOutp = np.where(feature[:,1] == outp)
                innerCoef = np.size(hasOutp)/np.size(feature, 0)
                featSum = p * (-innerCoef * math.log(innerCoef, 2))
                sum += featSum
                
        return sum

    def concatLabels(self, X, y):
        allData = np.concatenate((X, y), axis=1)
        return allData

    def printTree(self, node):
        if(node.level < 5):
            print("feature", node.feature, "val", node.value, "level", node.level)
        if(node.children == None): return
        for child in node.children:   
            self.printTree(child)

    #init call with node=head
    def splitData(self, X, y, node, labelsList):
        # print("-------X size", np.size(X,1))
        # print("Num Entries", np.size(y, 0))
        # print("labelsList", labelsList)
        # print("node.val", node.value, "nodeLevel", node.level)

        concatData = self.concatLabels(X, y)
        if(np.size(X, 1) == 1): 
            #final split and append targets (as feature)
            vals = np.unique(X)
            node.feature = labelsList[0]
            for val in vals:
                ind = np.where(X.flatten() == val)[0][0]
                target = y[int(ind),0]
                child = Node(node.level + 1, val, target, None) #level, val=None, feature=None, childs=[]):
                node.children.append(child)
            return
        gains = []
        #calculate which node to split on (min entropy)
        for i in range(np.size(X,1)):
            gains.append(self.getGains(X[:,i], y))
        minGainz = min(gains) #it says gains but it's acually entropy
        
        #split the data and build the tree
        colInd = gains.index(minGainz)
        #find colIndex in actual dataset
        colName = labelsList[colInd]
        node.feature = colName #do label of column
        childVals = self.featureCounts[self.featureLabels.index(colName)] 
        cutLabels = labelsList.copy()
        cutLabels.remove(colName)

        for val in range(childVals):
            sData = concatData[concatData[:,colInd] == val] #all vals in splitData with that val
            sData = np.delete(sData, colInd, 1) #then del col
            if(np.size(sData) == 0):#if there is no data for this va
                target = np.average(y)
                child = Node(node.level + 1, val, target, None) #level, val=None, feature=None, childs=[]):
                node.children.append(child)
            else: 
                childNode = Node(node.level + 1, val, None, [])
                node.children.append(childNode)
                self.splitData(sData[:,:sData.shape[1]-1], sData[:,sData.shape[1]-1:], childNode, cutLabels)
        return


    """ Predict all classes for a dataset X
    Args:
        X (array-like): A 2D numpy array with the training data, excluding targets
    Returns:
        array, shape (n_samples,)
            Predicted target values per element in X.
    """
    def predict(self, X):
        node = self.head
        pred = ""
        lastInd = ""
        while(node.children != None):
            colInd = self.featureLabels.index(node.feature)
            for child in node.children:
                if(lastInd == colInd):
                    return child.feature
                if X[colInd] == child.value:
                    pred = child.feature
                    node = child
                    break
            lastInd = colInd
        return pred

    """ Return accuracy of model on a given dataset. Must implement own score function.
    Args:
        X (array-like): A 2D numpy array with data, excluding targets
        y (array-li    def _shuffle_data(self, X, y):
    """
    def score(self, X, y):
        correctCount = 0
        totalCount = 0

        for inputs, exp in zip(X, y):
            pred = self.predict(inputs)
            if(exp == pred): correctCount += 1
            totalCount += 1

        self.accuracy = correctCount/totalCount
        #self.accArray.append(self.accuracy)
        #find num outputs that we got right
        return self.accuracy

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        xcols = X.shape[1]

        allData = np.concatenate((X, y), axis=1)
        np.random.shuffle(allData)
    
        y = allData[: , xcols:]
        X = allData[:, :xcols]
        return X, y