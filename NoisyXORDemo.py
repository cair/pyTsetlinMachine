#!/usr/local/bin/python3

from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np 
from time import time

train_data = np.loadtxt("data/NoisyXORTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("data/NoisyXORTestData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

ctm = MultiClassTsetlinMachine(10, 15, 3.9)

average = 0.0
for i in range(100):
	start = time()
	ctm.fit(X_train, Y_train, epochs=200)
	stop = time()

	result = ctm.evaluate(X_test, Y_test)
	
	average += result
	print(i+1, result, average/(i+1), stop-start)



