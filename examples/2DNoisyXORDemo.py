#!/usr/local/bin/python3

from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np 

train_data = np.loadtxt("2DNoisyXORTrainingData.txt")
X_train = train_data[:,0:-1].reshape(train_data.shape[0], 4, 4)
Y_train = train_data[:,-1]

test_data = np.loadtxt("2DNoisyXORTestData.txt")
X_test = test_data[:,0:-1].reshape(test_data.shape[0], 4, 4)
Y_test = test_data[:,-1]

ctm = MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2, 2), boost_true_positive_feedback=0)

ctm.fit(X_train, Y_train, epochs=5000)

print("Accuracy: %.2f%%" % (100.0*ctm.evaluate(X_test, Y_test)))
