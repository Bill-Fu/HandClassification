import numpy as np 
import cv2
import os
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation

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

###############################################################################

# train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(train_X,train_Y,test_size=0.3)
para_C = 0.01
clf = svm.LinearSVC(C=para_C)
clf.fit(train_X, train_Y)

print "Parameter C:" + str(para_C)
print "Accuracy of uniform testing data:" + str(clf.score(test_uniform_X, test_uniform_Y))
print "Accuracy of complex testing data:" + str(clf.score(test_complex_X, test_complex_Y))
# print "Accuracy of cross_validation data:" + str(clf.score(test_X, test_Y))

def Classify(name, desc, clf):
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    HOG = desc.compute(img)
    HOG = HOG.reshape((1,3780))
    print labels[clf.predict(HOG)[0]]
