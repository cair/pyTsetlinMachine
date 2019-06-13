#!/usr/local/bin/python3

from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine
import numpy as np 
from time import time

train_data = np.loadtxt("2DNoisyXORTrainingData.txt")
X_train = train_data[:,0:-1].reshape(train_data.shape[0], 4, 4)
Y_train = train_data[:,-1]

test_data = np.loadtxt("2DNoisyXORTestData.txt")
X_test = test_data[:,0:-1].reshape(test_data.shape[0], 4, 4)
Y_test = test_data[:,-1]

ctm = MultiClassConvolutionalTsetlinMachine(40, 60, 3.9, patch_dim=(2, 2))

results = np.zeros(0)
for i in range(100):
	start = time()
	ctm.fit(X_train, Y_train, epochs=5000)
	stop = time()

	results = np.append(results, 100.0*ctm.evaluate(X_test, Y_test))
	print("#%d Mean Accuracy (%%): %.2f; Std.dev.: %.2f; Training Time: %.1f ms/epoch" % (i+1, np.mean(results), np.std(results), (stop-start)/5.0))
