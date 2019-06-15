# pyTsetlinMachine

## Installation

```bash
pip install pyTsetlinMachine
```

## Examples

### Noisy XOR Demo

#### Code: NoisyXORDemo.py

```bash
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
print("Accuracy: %.2f%%" % (100.0*tm.evaluate(X_test, Y_test)))

print("Prediction: x1 = 1, x2 = 0, ... -> y = %d" % (tm.predict(np.array([1,0,1,0,1,0,1,1,1,1,0,0]))))
print("Prediction: x1 = 0, x2 = 1, ... -> y = %d" % (tm.predict(np.array([0,1,1,0,1,0,1,1,1,1,0,0]))))
print("Prediction: x1 = 0, x2 = 0, ... -> y = %d" % (tm.predict(np.array([0,0,1,0,1,0,1,1,1,1,0,0]))))
print("Prediction: x1 = 1, x2 = 1, ... -> y = %d" % (tm.predict(np.array([1,1,1,0,1,0,1,1,1,1,0,0]))))
```


#### Output

```bash
./NoisyXORDemo.py 

Accuracy: 100.00%
Prediction: x1 = 1, x2 = 0, ... -> y = 1
Prediction: x1 = 0, x2 = 1, ... -> y = 1
Prediction: x1 = 0, x2 = 0, ... -> y = 0
Prediction: x1 = 1, x2 = 1, ... -> y = 0
```

### 2D Noisy XOR Demo

#### Code: 2DNoisyXORDemo.py

```bash
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

Xi = np.array([[0,1,1,0],
		[1,1,0,1],
		[1,0,1,1],
		[0,0,0,1]])

print("\nInput Image:\n")
print(Xi)
print("\nPrediction: %d" % (ctm.predict(Xi)))
```

#### Output

```bash
./2DNoisyXORDemo.py 

Accuracy: 99.97%

Input Image:

[[0 1 1 0]
 [1 1 0 1]
 [1 0 1 1]
 [0 0 0 1]]

Prediction: 1
```
### MNIST Demo

Coming soon.

### Regression Demo

Coming soon.

## Tutorials

* Convolutional Tsetlin Machine tutorial, https://github.com/cair/convolutional-tsetlin-machine

## Further Work

* Binarization of continuous features using thresholding, https://arxiv.org/abs/1905.04199
* Multi-layer Tsetlin Machine
* Recurrent Tsetlin Machine

## Requirements

- Python 3.7.x, https://www.python.org/downloads/
- Numpy, http://www.numpy.org/

## Citation

```bash
@article{granmo2019convtsetlin,
  author = {{Granmo}, Ole-Christoffer and {Glimsdal}, Sondre and {Jiao}, Lei and {Goodwin}, Morten and {Omlin}, Christian W. and {Berge}, Geir Thore},
  title = "{The Convolutional Tsetlin Machine}",
  journal={arXiv preprint arXiv:1905.09688}, year={2019}
}
```

```bash
@article{abeyrathna2019regressiontsetlin,
  author = {{Abeyrathna}, Darshana and {Granmo}, Ole-Christoffer and {Jiao}, Lei and {Goodwin}, Morten},
  title = "{The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems}",
  journal={arXiv preprint arXiv:1905.04206}, year={2019}
}
```

```bash
@article{abeyrathna2019continuousinput,
  author = {{Abeyrathna}, Darshana and {Granmo}, Ole-Christoffer and {Zhang}, Xuan and {Goodwin}, Morten},
  title = "{A Scheme for Continuous Input to the Tsetlin Machine with Applications to Forecasting Disease Outbreaks}",
  journal={arXiv preprint arXiv:1905.04199}, year={2019}
}
```

```bash
@article{granmo2018tsetlin,
  author = {{Granmo}, Ole-Christoffer},
  title = "{The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic}",
  journal={arXiv preprint arXiv:1804.01508}, year={2018}
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
