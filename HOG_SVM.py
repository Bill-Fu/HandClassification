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

rootDir = "C:/Users/wb-fh265231/Desktop/dataset"
trainDir = "C:/Users/wb-fh265231/Desktop/dataset/train"
testDir = "C:/Users/wb-fh265231/Desktop/dataset/test"

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

# Get features and labels of training data
for i in range(len(labels)):
    for image in os.listdir(os.path.join(trainDir,labels[i])):
        train_Y.append(i)
        img = cv2.imread(os.path.join(trainDir,labels[i],image),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, win_size)
        HOG = desc.compute(img)
        train_X.append(HOG.tolist())
    # print str(i)

# Convert training data list to array
train_X = np.array(train_X)
train_Y = np.array(train_Y)

train_X = train_X.reshape((len(train_X),len(train_X[0])))

###############################################################################

# Get features and labels of uniform testing data
for i in range(len(labels)):
    for image in os.listdir(os.path.join(testDir,labels[i],"uniform")):
        test_uniform_Y.append(i)
        img = cv2.imread(os.path.join(testDir,labels[i],"uniform",image),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, win_size)
        HOG = desc.compute(img)
        test_uniform_X.append(HOG.tolist())
    # print str(i)

test_uniform_X = np.array(test_uniform_X)
test_uniform_Y = np.array(test_uniform_Y)

test_uniform_X = test_uniform_X.reshape((len(test_uniform_X),len(test_uniform_X[0])))

###############################################################################

# Get features and labels of complex testing data
for i in range(len(labels)):
    for image in os.listdir(os.path.join(testDir,labels[i],"complex")):
        test_complex_Y.append(i)
        img = cv2.imread(os.path.join(testDir,labels[i],"complex",image),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, win_size)
        HOG = desc.compute(img)
        test_complex_X.append(HOG.tolist())
    # print str(i)

test_complex_X = np.array(test_complex_X)
test_complex_Y = np.array(test_complex_Y)

test_complex_X = test_complex_X.reshape((len(test_complex_X),len(test_complex_X[0])))

print "Data for training load successfully"

###############################################################################

# train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(train_X,train_Y,test_size=0.3)
para_C = 0.007
clf_svm = svm.LinearSVC(C=para_C)
clf_svm.fit(train_X, train_Y)

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
"""
para_n_neighbors = 3
para_weights = 'distance'
clf_knn = neighbors.KNeighborsClassifier(para_n_neighbors, weights=para_weights)
clf_knn.fit(train_X, train_Y)

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

def Classify(name, desc, clf):
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    HOG = desc.compute(img)
    HOG = HOG.reshape((1,3780))
    print labels[clf.predict(HOG)[0]]
