from decisionTree import DTClassifier 
from arff import Arff
import numpy as np
import pandas as pd
import graph_tools
import string; 


mat = Arff("datasets/lenses.arff")

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

labelNames = [char for char in string.ascii_lowercase[:data.shape[1]]]

DTClass = DTClassifier(counts, labelNames)
DTClass.fit(data,labels)


# mat2 = Arff("/datasets/all_lenses.arff")
# data2 = mat2.data[:,0:-1]
# labels2 = mat2.data[:,-1]
# pred = DTClass.predict(data2)
# Acc = DTClass.score(data2,labels2)
# np.savetxt("pred_lenses.csv",pred,delimiter=",")
# print("Accuracy = [{:.2f}]".format(Acc))