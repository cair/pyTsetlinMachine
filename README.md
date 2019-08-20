# pyTsetlinMachine

Implementation of the Tsetlin Machine (https://arxiv.org/abs/1804.01508), Convolutional Tsetlin Machine (https://arxiv.org/abs/1905.09688) and Regression Tsetlin Machine (https://arxiv.org/abs/1905.04206), with support for continuous features (https://arxiv.org/abs/1905.04199, https://link.springer.com/chapter/10.1007%2F978-3-030-22999-3_49).

## Installation

```bash
pip install pyTsetlinMachine
```

## Tutorials

* Convolutional Tsetlin Machine tutorial, https://github.com/cair/convolutional-tsetlin-machine

## Examples

### Multiclass Demo

#### Code: NoisyXORDemo.py

```python
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np 

train_data = np.loadtxt("NoisyXORTrainingData.txt")
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("NoisyXORTestData.txt")
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

print("Prediction: x1 = 1, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[1,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[0,1,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[0,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 1, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[1,1,1,0,1,0,1,1,1,1,0,0]]))))
```

#### Output

```bash
python3 ./NoisyXORDemo.py 

Accuracy: 100.00%

Prediction: x1 = 1, x2 = 0, ... -> y = 1
Prediction: x1 = 0, x2 = 1, ... -> y = 1
Prediction: x1 = 0, x2 = 0, ... -> y = 0
Prediction: x1 = 1, x2 = 1, ... -> y = 0
```

### 2D Convolution Demo

#### Code: 2DNoisyXORDemo.py

```python
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

print("Accuracy:", 100*(ctm.predict(X_test) == Y_test).mean())

Xi = np.array([[[0,1,1,0],
		[1,1,0,1],
		[1,0,1,1],
		[0,0,0,1]]])

print("\nInput Image:\n")
print(Xi)
print("\nPrediction: %d" % (ctm.predict(Xi)))
```

#### Output

```bash
python3 ./2DNoisyXORDemo.py 

Accuracy: 99.97%

Input Image:

[[0 1 1 0]
 [1 1 0 1]
 [1 0 1 1]
 [0 0 0 1]]

Prediction: 1
```

### Continuous Input Demo

#### Code: BreastCancerDemo.py

```python
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from pyTsetlinMachine.tools import Binarizer
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target

b = Binarizer(max_bits_per_feature = 10)
b.fit(X)
X_transformed = b.transform(X)

tm = MultiClassTsetlinMachine(800, 40, 5.0)

print("\nMean accuracy over 100 runs:\n")
tm_results = np.empty(0)
for i in range(100):
	X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y, test_size=0.2)

	tm.fit(X_train, Y_train, epochs=25)
	tm_results = np.append(tm_results, np.array(100*(tm.predict(X_test) == Y_test).mean()))
	print("#%d Average Accuracy: %.2f%% +/- %.2f" % (i+1, tm_results.mean(), 1.96*tm_results.std()/np.sqrt(i+1)))
```
#### Output

```bash
python3 ./BreastCancerDemo.py 

Mean accuracy over 100 runs:

#1 Average Accuracy: 97.37% +/- 0.00
#2 Average Accuracy: 97.37% +/- 0.00
...
#99 Average Accuracy: 97.52% +/- 0.29
#100 Average Accuracy: 97.54% +/- 0.29
```

### MNIST Demo

#### Code: MNISTDemo.py

```python
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine(2000, 50, 10.0)

print("\nAccuracy over 400 epochs:\n")
for i in range(400):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
```

#### Output

```bash
python3 ./MNISTDemo.py 

Accuracy over 200 epochs:

#1 Accuracy: 94.57% (54.51s)
#2 Accuracy: 95.92% (44.28s)
#3 Accuracy: 96.28% (39.69s)
...

#198 Accuracy: 98.17% (29.46s)
#199 Accuracy: 98.19% (29.49s)
#200 Accuracy: 98.14% (29.49s)
```

### MNIST 2D Convolution Demo

#### Code: MNISTDemo2DConvolution.py

```python
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0) 

tm = MultiClassConvolutionalTsetlinMachine2D(8000, 200, 10.0, (10, 10))

print("\nAccuracy over 25 epochs:\n")
for i in range(25):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	
	result = 100*(tm.predict(X_test) == Y_test).mean()
	
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
```

#### Output

```bash
python3 ./MNISTDemo2DConvolution.py 

Accuracy over 40 epochs:

#1 Accuracy: 97.81% (1383.61s)
#2 Accuracy: 98.42% (1383.16s)
#3 Accuracy: 98.52% (1387.48s)
...

#38 Accuracy: 99.11% (1381.82s)
#39 Accuracy: 99.07% (1225.61s)
#40 Accuracy: 99.13% (1379.31s)
```

### IMDb Text Categorization Demo

#### Code: IMDbTextCategorizationDemo.py

```python
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from time import time

MAX_NGRAM = 2

NUM_WORDS=5000
INDEX_FROM=2 

FEATURES=5000

print("Downloading dataset...")

# Save np.load
np_load_old = np.load

# Modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

# Restore np.load for future normal usage
np.load = np_load_old

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

# Produce N-grams

id_to_word = {value:key for key,value in word_to_id.items()}

vocabulary = {}
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])
	
	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			
			if phrase in vocabulary:
				vocabulary[phrase] += 1
			else:
				vocabulary[phrase] = 1

# Assign a bit position to each N-gram (minimum frequency 10) 

phrase_bit_nr = {}
bit_nr_phrase = {}
bit_nr = 0
for phrase in vocabulary.keys():
	if vocabulary[phrase] < 10:
		continue

	phrase_bit_nr[phrase] = bit_nr
	bit_nr_phrase[bit_nr] = phrase
	bit_nr += 1

# Create bit representation

X_train = np.zeros((train_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_train[i,phrase_bit_nr[phrase]] = 1

	Y_train[i] = train_y[i]

X_test = np.zeros((test_y.shape[0], len(phrase_bit_nr)), dtype=np.uint32)
Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)

for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id])

	for N in range(1,MAX_NGRAM+1):
		grams = [terms[j:j+N] for j in range(len(terms)-N+1)]
		for gram in grams:
			phrase = " ".join(gram)
			if phrase in phrase_bit_nr:
				X_test[i,phrase_bit_nr[phrase]] = 1				

	Y_test[i] = test_y[i]

print("Selecting features...")

SKB = SelectKBest(chi2, k=FEATURES)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train)
X_test = SKB.transform(X_test)

tm = MultiClassTsetlinMachine(10000, 80, 27.0)

print("\nAccuracy over 25 epochs:\n")
for i in range(25):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
```

#### Output:

```bash
python ./IMDbTextCategorizationDemo.py

Downloading dataset...
Producing bit representation...
Selecting features...

Accuracy over 25 epochs:

#1 Accuracy: 87.10% (1129.02s)
#2 Accuracy: 87.72% (1136.42s)
#3 Accuracy: 88.08% (1076.66s)
...

#23 Accuracy: 89.36% (733.84s)
#24 Accuracy: 89.34% (736.17s)
#25 Accuracy: 89.36% (826.29s)
```

### Regression Demo

#### Code: RegressionDemo.py

```python
from pyTsetlinMachine.tm import RegressionTsetlinMachine
from pyTsetlinMachine.tools import Binarizer
import numpy as np
from time import time

from sklearn import datasets
from sklearn.model_selection import train_test_split

california_housing = datasets.fetch_california_housing()
X = california_housing.data
Y = california_housing.target

b = Binarizer(max_bits_per_feature = 10)
b.fit(X)
X_transformed = b.transform(X)

tm = RegressionTsetlinMachine(4000, 2000, 2.75)

print("\nRMSD over 25 runs:\n")
tm_results = np.empty(0)
for i in range(25):
	X_train, X_test, Y_train, Y_test = train_test_split(X_transformed, Y)

	start = time()
	tm.fit(X_train, Y_train, epochs=30)
	stop = time()
	tm_results = np.append(tm_results, np.sqrt(((tm.predict(X_test) - Y_test)**2).mean()))

	print("#%d RMSD: %.2f +/- %.2f (%.2fs)" % (i+1, tm_results.mean(), 1.96*tm_results.std()/np.sqrt(i+1), stop-start))
```

#### Output

```bash
python3 ./RegressionDemo.py 

RMSD over 25 runs:

#1 RMSD: 0.62 +/- 0.00 (56.62s)
#2 RMSD: 0.60 +/- 0.02 (58.57s)
...

#24 RMSD: 0.61 +/- 0.00 (60.20s)
#25 RMSD: 0.61 +/- 0.00 (59.68s)
```

## Further Work

* Multilayer Tsetlin Machine
* Recurrent Tsetlin Machine
* GPU support
* Multi-threading
* Optimize convolution code
* More extensive hyper-parameter search for the demos

## Requirements

- Python 3.7.x, https://www.python.org/downloads/
- Numpy, http://www.numpy.org/
- Ubuntu or macOS

## Citation

```bash
@article{granmo2019convtsetlin,
  author = {{Granmo}, Ole-Christoffer and {Glimsdal}, Sondre and {Jiao}, Lei and {Goodwin}, Morten and {Omlin}, Christian W. and {Berge}, Geir Thore},
  title = "{The Convolutional Tsetlin Machine}",
  journal = {arXiv preprint arXiv:1905.09688}, year = {2019}
}
```

```bash
@article{abeyrathna2019regressiontsetlin,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Jiao}, Lei and {Goodwin}, Morten},
  title = "{The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems}",
  journal = {arXiv preprint arXiv:1905.04206}, year = {2019}
}
```

```bash
@InProceedings{abeyrathna2019continuousinput,
  author = {{Abeyrathna}, Kuruge Darshana and {Granmo}, Ole-Christoffer and {Zhang}, Xuan and {Goodwin}, Morten},
  title = "{A Scheme for Continuous Input to the Tsetlin Machine with Applications to Forecasting Disease Outbreaks}",
  booktitle = "{Advances and Trends in Artificial Intelligence. From Theory to Practice}", year = "2019",
  editor = "Wotawa, Franz and Friedrich, Gerhard and Pill, Ingo and Koitz-Hristov, Roxane and Ali, Moonis",
  publisher = "Springer International Publishing",
  pages = "564--578"
}
```

```bash
@article{granmo2018tsetlin,
  author = {{Granmo}, Ole-Christoffer},
  title = "{The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic}",
  journal = {arXiv preprint arXiv:1804.01508}, year = {2018}
}
```

## Licence

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
