from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np 

X_train = np.random.randint(0, 2, size=(5000, 12), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000) < 0.1, 1-Y_train, Y_train) # Adds 10% noise

X_test = np.random.randint(0, 2, size=(5000, 12), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

tm = MultiClassTsetlinMachine(10, 15, 3.0, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("\nClass 0 Positive Clauses:\n")
for j in range(0, 10, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(24):
		if tm.ta_action(0, j, k) == 1:
			if k < 12:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-12))
	print(" ∧ ".join(l))

print("\nClass 1 Positive Clauses:\n")
for j in range(0, 10, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(24):
		if tm.ta_action(1, j, k) == 1:
			if k < 12:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-12))
	print(" ∧ ".join(l))
