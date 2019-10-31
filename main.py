from decisionTree import DTClassifier 
from arff import Arff
import numpy as np
import pandas as pd
import graph_tools
import string
import math

import pydotplus
from sklearn import tree
import collections
from graphviz import Source
from IPython.display import Image


mat = Arff("datasets/tictactoe.arff")

counts = [] ## this is so you know how many types for each column
for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

labelNames =  mat.get_attr_names() #[char for char in string.ascii_lowercase[:data.shape[1]]]
#del labelNames[-1]

DTSClass = DTClassifier(counts, labelNames, shuffle=True)

##########10 fold CV#########################
# # #split into 10 groups
# sData = np.array_split(data, 10, 0)
# sLabels = np.array_split(labels, 10, 0)
# accs = []

# # clf = tree.DecisionTreeClassifier(min_impurity_decrease=2)

# for i in range(1):
#        #print(i, inputs[i])
#        vData = np.copy(sData[i])
#        vLabels = np.copy(sLabels[i]) 

#        cData = list.copy(sData)
#        cLabels = list.copy(sLabels)
#        del cData[i]
#        del cLabels[i]

#        tData = np.concatenate(np.copy(cData[0:8], 0))
#        tLabels = np.concatenate(np.copy(cLabels[0:8], 0))

#        #Sklearn
#        # model = clf.fit(tData, tLabels)
#        # acc = clf.score(vData, vLabels)

#        #My DT
#        DTSClass.fit(tData, tLabels)
#        acc = DTSClass.score(vData, vLabels)

#        accs.append(acc)
# print("ACCURACIES")
# print(accs)
# print("AVG ACC", sum(accs)/len(accs))



###########Just run it normal#####################
# DTClass = DTClassifier(counts, labelNames)
# DTClass.fit(data,labels)
# acc = DTClass.score(data, labels)
# print("--ACCURACY 1--", acc)

# mat2 = Arff("datasets/all_lenses.arff")
# data2 = mat2.data[:,0:-1]
# labels2 = mat2.data[:,-1]

# preds = []
# for inputs in data2:
#        pred = DTClass.predict(inputs)
#        preds.append(pred)
# np.savetxt("idk.csv",preds,delimiter=",")

# Acc = DTClass.score(data2,labels2)
# print("Accuracy = [{:.2f}]".format(Acc))

################Sklearn stuff######################
clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=5)
model = clf.fit(data, labels)
print("SKLEARN ACC:", clf.score(data, labels))


#Graph stuff
featureNames = mat.get_attr_names()
del featureNames[-1]
graph = Source( tree.export_graphviz(model, feature_names=featureNames))
png_bytes = graph.pipe(format='png')
with open('tictac_tree.png','wb') as f:
    f.write(png_bytes)

Image(png_bytes)