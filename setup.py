from setuptools import *

libTM = Extension('libTM',
                  sources = ['pyTsetlinMachine/ConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/MultiClassConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/Tools.c'],
                  include_dirs=['pyTsetlinMachine'])

setup(
   name='pyTsetlinMachine',
   version='0.2.5',
   author='Ole-Christoffer Granmo',
   author_email='ole.granmo@uia.no',
   url='https://github.com/cair/pyTsetlinMachine/',
   license='MIT',
   description='Implements the Tsetlin Machine, Convolutional Tsetlin Machine and Regression Tsetlin Machine, , with support for continuous features.',
   long_description='Implements the Tsetlin Machine, Convolutional Tsetlin Machine and Regression Tsetlin Machine, with support for continuous features.',
   ext_modules = [libTM],
   keywords ='pattern-recognition machine-learning interpretable-machine-learning rule-based-machine-learning propositional-logic tsetlin-machine regression convolution',
   packages=['pyTsetlinMachine']
)
