import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class Node():
    def __init__(self, level, val="", col="", childs=[]):
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
        self.head = Node(0)

        self.g = 0


    """ Fit the data; Make the Desicion tree
    Args:
        X (array-like): A 2D numpy array with the training data, excluding targets
        y (array-like): A 2D numpy array with the training targets
    Returns:
        self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
    """
    def fit(self, X, y):
        self.splitData(X, y, self.head, self.featureLabels.copy())

        return self

    def getGains(self, array, labels):
        arySize = np.size(labels)
        allData = self.concatLabels(array.reshape(-1,1), labels.reshape(-1,1))
        vals = np.unique(allData[:,0])
        print("allData", allData)
        print("vals", vals)
        splits = []
        for i in range(np.size(vals)):
            splits.append(allData[allData[:,0] == vals[i]])
            #print("index", index)
            # indexes.append(index)
            # props.append(np.size(index)/arySize)
        # print("props", props)
        print("splits", splits)
        outps = np.unique(labels)
        sum = 0
        for outp in outps:
            ha = 1
        
        return 1

    def concatLabels(self, X, y):
        allData = np.concatenate((X, y), axis=1)
        return allData

    #init call with node=head
    def splitData(self, X, y, node, labelsList):
        print("-------X size", np.size(X,1))
        print("Num Entries", np.size(y, 0))
        print("labelsList", labelsList)
        print("node.val", node.value, "nodeLevel", node.level)
        concatData = self.concatLabels(X, y)
        gains = []
        if(np.size(X, 1) == 1): 
            return
        #calculate which node to split on (max gain)
        gains.clear()
        
        for i in range(np.size(X,1)):
            gains.append(self.getGains(X[:,i], y))
        maxGainz = max(gains)
        
        #split the data and build the tree
        colInd = gains.index(maxGainz)
        print("colInd", colInd)
        #find colIndex in actual dataset
        colName = labelsList[colInd]
        print("col to delete", colName)
        node.feature = colName #do label of column
        childVals = np.unique(X[:,colInd])
        labelsList.remove(colName)

        for val in childVals:
            sData = concatData[concatData[:,colInd] == val] #all vals in splitData with that val
            sData = np.delete(sData, colInd, 1) #then del col
            childNode = Node(node.level + 1, val, None, [])
            node.children.append(childNode)
            self.splitData(sData[:,:sData.shape[1]-1], sData[:,sData.shape[1]-1:], childNode, labelsList)
        return


    """ Predict all classes for a dataset X
    Args:
        X (array-like): A 2D numpy array with the training data, excluding targets
    Returns:
        array, shape (n_samples,)
            Predicted target values per element in X.
    """
    def predict(self, X):

        pass


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        return 0

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