#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys, os
from time import time

curDirectory = os.path.dirname(os.path.abspath(__file__) )

parDirectory = os.path.abspath(os.path.join(curDirectory, os.pardir))

print(parDirectory)

#sys.path.append("../tools/")
sys.path.append(parDirectory + '/tools')

#print('\n\n')
#print( sys.path )
#print('\n\n')

#sys.path.insert(0, "../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

t0 = time()
classifier.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = classifier.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"

print(pred)

t2 = time()
print "accuracy: ", classifier.score(features_test, labels_test)
print "scoring time:", round(time()-t2, 3), "s"

#########################################################


