from setuptools import *

#Windows throws 'LINK : error LNK2001: unresolved external symbol PyInit_libTM' error
def get_export_symbols(self, ext):
    parts = ext.name.split(".")
    print('parts', parts)
    if parts[-1] == "__init__":
        initfunc_name = "PyInit_" + parts[-2]
    else:
        initfunc_name = "PyInit_" + parts[-1]

libTM = Extension('libTM',
                  sources = ['pyTsetlinMachine/ConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/MultiClassConvolutionalTsetlinMachine.c', 'pyTsetlinMachine/Tools.c'],
                  include_dirs=['pyTsetlinMachine'])

setup(
   name='pyTsetlinMachine',
   version='0.2.0',
   author='Ole-Christoffer Granmo',
   author_email='ole.granmo@uia.no',
   url='https://github.com/cair/pyTsetlinMachine/',
   license='MIT',
   description='Implements the Tsetlin Machine, Convolutional Tsetlin Machine and Regression Tsetlin Machine',
   long_description='Implements the Tsetlin Machine, Convolutional Tsetlin Machine and Regression Tsetlin Machine',
   ext_modules = [libTM],
   keywords ='pattern-recognition machine-learning interpretable rule-based propositional-logic tsetlin-machine regression convolution',
   packages=['pyTsetlinMachine']
)
