from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np 

train_data = np.loadtxt("NoisyXORTrainingData.txt")
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("NoisyXORTestData.txt")
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(10, 15, 3.9)

tm.fit(X_train, Y_train, epochs=200)
print("Accuracy: %.2f%%" % (100.0*tm.evaluate(X_test, Y_test)))

print("Prediction: x1 = 1, x2 = 0, ... -> y = %d" % (tm.predict(np.array([1,0,1,0,1,0,1,1,1,1,0,0]))))
print("Prediction: x1 = 0, x2 = 1, ... -> y = %d" % (tm.predict(np.array([0,1,1,0,1,0,1,1,1,1,0,0]))))
print("Prediction: x1 = 0, x2 = 0, ... -> y = %d" % (tm.predict(np.array([0,0,1,0,1,0,1,1,1,1,0,0]))))
print("Prediction: x1 = 1, x2 = 1, ... -> y = %d" % (tm.predict(np.array([1,1,1,0,1,0,1,1,1,1,0,0]))))
