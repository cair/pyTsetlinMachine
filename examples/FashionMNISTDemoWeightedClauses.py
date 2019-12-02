from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import cv2

from keras.datasets import *


# train_data = np.loadtxt("FashionTrainingData.txt")
# X_train = train_data[:,0:-1]
# Y_train = train_data[:,-1]

# test_data = np.loadtxt("FashionTestData.txt")
# X_test = test_data[:,0:-1]
# Y_test = test_data[:,-1]

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = np.copy(X_train)
X_test = np.copy(X_test)

for i in range(X_train.shape[0]):
	X_train[i,:] = cv2.adaptiveThreshold(X_train[i],1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
X_train = X_train.reshape((X_train.shape[0], 28*28))

for i in range(X_test.shape[0]):
	X_test[i,:] = cv2.adaptiveThreshold(X_test[i],1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
X_test = X_test.reshape((X_test.shape[0], 28*28))

tm = MultiClassTsetlinMachine(2000, 50*100, 10.0, weighted_clauses=True)

print("\nAccuracy over 60 epochs:\n")
for i in range(250):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
