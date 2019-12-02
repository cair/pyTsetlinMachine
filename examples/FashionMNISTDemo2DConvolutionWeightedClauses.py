from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import cv2

from keras.datasets import *
import sys

clauses = int(sys.argv[1])
s = float(sys.argv[2])
T = int (sys.argv[3])

print(clauses, s, T)

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = np.copy(X_train)
X_test = np.copy(X_test)

for i in range(X_train.shape[0]):
	X_train[i,:] = cv2.adaptiveThreshold(X_train[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

for i in range(X_test.shape[0]):
	X_test[i,:] = cv2.adaptiveThreshold(X_test[i], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (10, 10), weighted_clauses=True)

f = open("stats_%d_%d_%.2f.txt" % (clauses, T, s), "w+")

print("\nAccuracy over 150 epochs:\n")
for i in range(150):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	
	result = 100*(tm.predict(X_test) == Y_test).mean()
	
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
	
	f.write("%d %.2f\n" % (i, result))
	f.flush()

f.close()