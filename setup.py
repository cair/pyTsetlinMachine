from setuptools import *

libTM = Extension('libTM',
  ['pyTsetlinMachine/ConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/MultiClassConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/Tools.c'])

setup(
   name='pyTsetlinMachine',
   version='0.1.0',
   author='Ole-Christoffer Granmo',
   author_email='ole.granmo@uia.no',
   url='http://pypi.python.org/pypi/pyTsetlinMachine/',
   license='MIT',
   description='Implements the Tsetlin Machine, Convolutional Tsetlin Machine, and Regression Tsetlin Machine',
   ext_modules = [libTM],
   packages=['pyTsetlinMachine'],
)
