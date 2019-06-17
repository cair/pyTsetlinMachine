from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0) 
X_test = np.where(X_test > 75, 1, 0) 

tm = MultiClassConvolutionalTsetlinMachine2D(2000, 50, 10.0, (10, 10))

print("\nAccuracy over 10 epochs:\n")
for i in range(10):
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	start = time()
	tm_results = 100*(tm.predict(X_test) == Y_test).mean()
	stop = time()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, tm_results, stop-start))
