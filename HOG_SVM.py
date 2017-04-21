#coding=utf-8

import numpy as np 
import cv2
import os
from sklearn import svm
from sklearn import neighbors
from sklearn import metrics
# 导入交叉验证库
from sklearn import cross_validation
# 生成预测结果准确率的混淆矩阵
from sklearn import metrics
# 导入Libsvm的python库
from svmutil import *

rootDir = "C:/Users/wb-fh265231/Dropbox/graduation_project/dataset_clean"
trainDir = "C:/Users/wb-fh265231/Dropbox/graduation_project/dataset_clean/train"
testDir = "C:/Users/wb-fh265231/Dropbox/graduation_project/dataset_clean/test"

win_size = (64, 128)

train_X=[]
train_Y=[]
test_complex_X=[]
test_complex_Y=[]
test_uniform_X=[]
test_uniform_Y=[]

labels = os.listdir(trainDir)

# HOGDescriptor in OpenCV
desc = cv2.HOGDescriptor()

###############################################################################
############################# Training Data ###################################
###############################################################################

# Get features and labels of training data
for i in range(len(labels)):
    for image in os.listdir(os.path.join(trainDir,labels[i])):
        train_Y.append(i)
        img = cv2.imread(os.path.join(trainDir,labels[i],image),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, win_size)
        HOG = desc.compute(img)
        HOG = HOG.reshape((len(HOG),))
        train_X.append(HOG.tolist())
    # print str(i)

# training data for Libsvm
train_X_libsvm = train_X
train_Y_libsvm = train_Y

# Convert training data from list to array
train_X = np.array(train_X)
train_Y = np.array(train_Y)

# train_X = train_X.reshape((len(train_X),len(train_X[0])))

###############################################################################
########################## Uniform Testing Data ###############################
###############################################################################

# Get features and labels of uniform testing data
for i in range(len(labels)):
    for image in os.listdir(os.path.join(testDir,labels[i],"uniform")):
        test_uniform_Y.append(i)
        img = cv2.imread(os.path.join(testDir,labels[i],"uniform",image),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, win_size)
        HOG = desc.compute(img)
        HOG = HOG.reshape((len(HOG),))
        test_uniform_X.append(HOG.tolist())
    # print str(i)

# uniform testing data for Libsvm
test_uniform_X_libsvm = test_uniform_X
test_uniform_Y_libsvm = test_uniform_Y

# Convert uniform testing data from list to array
test_uniform_X = np.array(test_uniform_X)
test_uniform_Y = np.array(test_uniform_Y)

# test_uniform_X = test_uniform_X.reshape((len(test_uniform_X),len(test_uniform_X[0])))

###############################################################################
######################## Complex Testing Data #################################
###############################################################################

# Get features and labels of complex testing data
for i in range(len(labels)):
    for image in os.listdir(os.path.join(testDir,labels[i],"complex")):
        test_complex_Y.append(i)
        img = cv2.imread(os.path.join(testDir,labels[i],"complex",image),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, win_size)
        HOG = desc.compute(img)
        HOG = HOG.reshape((len(HOG),))
        test_complex_X.append(HOG.tolist())
    # print str(i)

# complex testing data for Libsvm
test_complex_X_libsvm = test_complex_X
test_complex_Y_libsvm = test_complex_Y

# Convert complex testing data from list to array
test_complex_X = np.array(test_complex_X)
test_complex_Y = np.array(test_complex_Y)

# test_complex_X = test_complex_X.reshape((len(test_complex_X),len(test_complex_X[0])))

print "Data for training load successfully\n"
"""
###############################################################################
############## Scikit-learn SVM Training and Testing ##########################
###############################################################################

# train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(train_X,train_Y,test_size=0.3)
para_C = 0.007
clf_svm = svm.LinearSVC(C=para_C)
clf_svm.fit(train_X, train_Y)

print "SVM training result:"
print "Parameter C:" + str(para_C)
print "Accuracy of uniform testing data:" + str(clf_svm.score(test_uniform_X, test_uniform_Y))
print "Accuracy of complex testing data:" + str(clf_svm.score(test_complex_X, test_complex_Y))
# print "Accuracy of cross_validation data:" + str(clf.score(test_X, test_Y))

predicted_uniform = clf_svm.predict(test_uniform_X)
expected_uniform = test_uniform_Y

predicted_complex = clf_svm.predict(test_complex_X)
expected_complex = test_complex_Y

print "Confusion matrix of uniform testing data"
print metrics.confusion_matrix(expected_uniform, predicted_uniform)
print "Confusion matrix of complex testing data"
print metrics.confusion_matrix(expected_complex, predicted_complex)

###############################################################################
################ Scikit-learn kNN Training and Testing ########################
###############################################################################

para_n_neighbors = 11
para_weights = 'distance'
clf_knn = neighbors.KNeighborsClassifier(para_n_neighbors, weights=para_weights)
clf_knn.fit(train_X, train_Y)

print "\nkNN training result:"
print "Parameter k:" + str(para_n_neighbors)
print "Parameter weights:" + str(para_weights)
print "Accuracy of uniform testing data:" + str(clf_knn.score(test_uniform_X, test_uniform_Y))
print "Accuracy of complex testing data:" + str(clf_knn.score(test_complex_X, test_complex_Y))

predicted_uniform_knn = clf_knn.predict(test_uniform_X)
expected_uniform_knn = test_uniform_Y

predicted_complex_knn = clf_knn.predict(test_complex_X)
expected_complex_knn = test_complex_Y

print "Confusion matrix of uniform testing data"
print metrics.confusion_matrix(expected_uniform_knn, predicted_uniform_knn)
print "Confusion matrix of complex testing data"
print metrics.confusion_matrix(expected_complex_knn, predicted_complex_knn)
"""
###############################################################################
############### Libsvm SVM Training, Testing and Model Saving #################
###############################################################################

param_mod = '-t 0 -c 0.0105 -b 1 -m 1000'
param_pred = '-b 1'
prob = svm_problem(train_Y_libsvm, train_X_libsvm)
param = svm_parameter(param_mod)
model = svm_train(prob, param)
p_labs_uniform, p_acc_uniform, p_vals_uniform = svm_predict(test_uniform_Y_libsvm, test_uniform_X_libsvm, model, param_pred)
p_labs_complex, p_acc_complex, p_vals_complex = svm_predict(test_complex_Y_libsvm, test_complex_X_libsvm, model, param_pred)

print "Model parameters:" + param_mod
print "Prediction parameters:" + param_pred
print "Accuracy of uniform testing data:" + str(p_acc_uniform[0])
print "Accuracy of complex testing data:" + str(p_acc_complex[0])

svm_save_model('handGestureClassificationModel', model)

def Classify(name, desc, clf):
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    HOG = desc.compute(img)
    HOG = HOG.reshape((1,3780))
    print labels[clf.predict(HOG)[0]]
