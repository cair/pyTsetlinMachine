# Copyright (c) 2019 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import numpy as np
import ctypes as C
import os

this_dir, this_filename = os.path.split(__file__)
_lib = np.ctypeslib.load_library('libTM', os.path.join(this_dir, ".."))    

class CMultiClassConvolutionalTsetlinMachine(C.Structure):
	None

class CConvolutionalTsetlinMachine(C.Structure):
	None

class CIndexedTsetlinMachine(C.Structure):
	None

mc_ctm_pointer = C.POINTER(CMultiClassConvolutionalTsetlinMachine)

ctm_pointer = C.POINTER(CConvolutionalTsetlinMachine)

itm_pointer = C.POINTER(CIndexedTsetlinMachine)

array_1d_uint = np.ctypeslib.ndpointer(
	dtype=np.uint32,
	ndim=1,
	flags='CONTIGUOUS')

array_1d_int = np.ctypeslib.ndpointer(
	dtype=np.int32,
	ndim=1,
	flags='CONTIGUOUS')


# Multiclass Tsetlin Machine

_lib.CreateMultiClassTsetlinMachine.restype = mc_ctm_pointer                    
_lib.CreateMultiClassTsetlinMachine.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double, C.c_int, C.c_int] 

_lib.mc_tm_destroy.restype = None                      
_lib.mc_tm_destroy.argtypes = [mc_ctm_pointer] 

_lib.mc_tm_fit.restype = None                      
_lib.mc_tm_fit.argtypes = [mc_ctm_pointer, array_1d_uint, array_1d_uint, C.c_int, C.c_int] 

_lib.mc_tm_initialize.restype = None                      
_lib.mc_tm_initialize.argtypes = [mc_ctm_pointer] 

_lib.mc_tm_predict.restype = None                    
_lib.mc_tm_predict.argtypes = [mc_ctm_pointer, array_1d_uint, array_1d_uint, C.c_int] 

_lib.mc_tm_ta_state.restype = C.c_int                    
_lib.mc_tm_ta_state.argtypes = [mc_ctm_pointer, C.c_int, C.c_int, C.c_int]

_lib.mc_tm_ta_action.restype = C.c_int                    
_lib.mc_tm_ta_action.argtypes = [mc_ctm_pointer, C.c_int, C.c_int, C.c_int] 

_lib.mc_tm_set_state.restype = None
_lib.mc_tm_set_state.argtypes = [mc_ctm_pointer, C.c_int, array_1d_uint]

_lib.mc_tm_get_state.restype = None
_lib.mc_tm_get_state.argtypes = [mc_ctm_pointer, C.c_int, array_1d_uint]

_lib.mc_tm_transform.restype = None                    
_lib.mc_tm_transform.argtypes = [mc_ctm_pointer, array_1d_uint, array_1d_uint, C.c_int, C.c_int] 

_lib.mc_tm_clause_configuration.restype = None                    
_lib.mc_tm_clause_configuration.argtypes = [mc_ctm_pointer, C.c_int, C.c_int, array_1d_uint] 

# Tsetlin Machine

_lib.CreateTsetlinMachine.restype = ctm_pointer                    
_lib.CreateTsetlinMachine.argtypes = [C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_double, C.c_double, C.c_int, C.c_int] 

_lib.tm_fit_regression.restype = None                      
_lib.tm_fit_regression.argtypes = [ctm_pointer, array_1d_uint, array_1d_int, C.c_int, C.c_int] 

_lib.tm_predict_regression.restype = None                    
_lib.tm_predict_regression.argtypes = [ctm_pointer, array_1d_uint, array_1d_int, C.c_int] 

# Tools

_lib.tm_encode.restype = None                      
_lib.tm_encode.argtypes = [array_1d_uint, array_1d_uint, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int] 

# Indexed Tsetlin Machine

_lib.CreateIndexedTsetlinMachine.restype = itm_pointer
_lib.CreateIndexedTsetlinMachine.argtypes = [mc_ctm_pointer]

_lib.itm_destroy.restype = None                      
_lib.itm_destroy.argtypes = [itm_pointer] 

_lib.itm_initialize.restype = None
_lib.itm_initialize.argtypes = [itm_pointer]

_lib.itm_predict.restype = None                    
_lib.itm_predict.argtypes = [itm_pointer, array_1d_uint, array_1d_uint, C.c_int]

_lib.itm_fit.restype = None                      
_lib.itm_fit.argtypes = [itm_pointer, array_1d_uint, array_1d_uint, C.c_int, C.c_int]

_lib.itm_transform.restype = None                    
_lib.itm_transform.argtypes = [itm_pointer, array_1d_uint, array_1d_uint, C.c_int, C.c_int]

class MultiClassConvolutionalTsetlinMachine2D():
	def __init__(self, number_of_clauses, T, s, patch_dim, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, weighted_clauses=False, s_range=False):
		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)/32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.patch_dim = patch_dim
		self.T = int(T)
		self.s = s
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.mc_ctm = None
		self.append_negated = append_negated
		self.weighted_clauses = weighted_clauses
		if s_range:
			self.s_range = s_range
		else:
			self.s_range = s

	def __del__(self):
		if self.mc_ctm != None:
			_lib.mc_tm_destroy(self.mc_ctm)

	def fit(self, X, Y, epochs=100, incremental=False):
		number_of_examples = X.shape[0]

		if self.mc_ctm == None:
			self.number_of_classes = np.unique(Y).size
			self.dim_x = X.shape[1]
			self.dim_y = X.shape[2]

			if len(X.shape) == 3:
				self.dim_z = 1
			elif len(X.shape) == 4:
				self.dim_z = X.shape[3]

			if self.append_negated:
				self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim_z + (self.dim_x - self.patch_dim[0]) + (self.dim_y - self.patch_dim[1]))*2
			else:
				self.number_of_features = int(self.patch_dim[0]*self.patch_dim[1]*self.dim_z + (self.dim_x - self.patch_dim[0]) + (self.dim_y - self.patch_dim[1]))

			self.number_of_patches = int((self.dim_x - self.patch_dim[0] + 1)*(self.dim_y - self.patch_dim[1] + 1))
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
			self.mc_ctm = _lib.CreateMultiClassTsetlinMachine(self.number_of_classes, self.number_of_clauses, self.number_of_features, self.number_of_patches, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)
		elif incremental == False:
			_lib.mc_tm_destroy(self.mc_ctm)
			self.mc_ctm = _lib.CreateMultiClassTsetlinMachine(self.number_of_classes, self.number_of_clauses, self.number_of_features, self.number_of_patches, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)

		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		if self.append_negated:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.dim_x, self.dim_y, self.dim_z, self.patch_dim[0], self.patch_dim[1], 1)
		else:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.dim_x, self.dim_y, self.dim_z, self.patch_dim[0], self.patch_dim[1], 0)
		
		_lib.mc_tm_fit(self.mc_ctm, self.encoded_X, Ym, number_of_examples, epochs)

		return

	def predict(self, X):
		number_of_examples = X.shape[0]
		
		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)

		if self.append_negated:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.dim_x, self.dim_y, self.dim_z, self.patch_dim[0], self.patch_dim[1], 1)
		else:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.dim_x, self.dim_y, self.dim_z, self.patch_dim[0], self.patch_dim[1], 0)
	
		Y = np.ascontiguousarray(np.zeros(number_of_examples, dtype=np.uint32))

		_lib.mc_tm_predict(self.mc_ctm, self.encoded_X, Y, number_of_examples)

		return Y

	def ta_state(self, mc_tm_class, clause, ta):
		return _lib.mc_tm_ta_state(self.mc_ctm, mc_tm_class, clause, ta)

	def ta_action(self, mc_tm_class, clause, ta):
		return _lib.mc_tm_ta_action(self.mc_ctm, mc_tm_class, clause, ta)

	def get_state(self):
		state_list = []
		for i in range(self.number_of_classes):
			ta_states = np.ascontiguousarray(np.empty(self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits, dtype=np.uint32))
			_lib.mc_tm_get_state(self.mc_ctm, i, ta_states)
			state_list.append(ta_states)

		return state_list

	def set_state(self, ta_states):
		for i in range(self.number_of_classes):
			_lib.mc_tm_set_state(self.mc_ctm, i, ta_states[i])

		return

	def transform(self, X, inverted=True):
		number_of_examples = X.shape[0]

		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))
		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)

		if self.append_negated:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.dim_x, self.dim_y, self.dim_z, self.patch_dim[0], self.patch_dim[1], 1)
		else:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.dim_x, self.dim_y, self.dim_z, self.patch_dim[0], self.patch_dim[1], 0)

		X_transformed = np.ascontiguousarray(np.empty(number_of_examples*self.number_of_classes*self.number_of_clauses, dtype=np.uint32))
		
		if (inverted):
			_lib.mc_tm_transform(self.mc_ctm, self.encoded_X, X_transformed, 1, number_of_examples)
		else:
			_lib.mc_tm_transform(self.mc_ctm, self.encoded_X, X_transformed, 0, number_of_examples)
		
		return X_transformed.reshape((number_of_examples, self.number_of_classes*self.number_of_clauses))

class MultiClassTsetlinMachine():
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, indexed=True, append_negated=True, weighted_clauses=False, s_range=False):
		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)/32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.mc_tm = None
		self.itm = None
		self.indexed = indexed
		self.append_negated = append_negated
		self.weighted_clauses = weighted_clauses
		if s_range:
			self.s_range = s_range
		else:
			self.s_range = s

	def __del__(self):
		if self.mc_tm != None:
			_lib.mc_tm_destroy(self.mc_tm)

		if self.itm != None:
			_lib.itm_destroy(self.itm)

	def fit(self, X, Y, epochs=100, incremental=False):
		number_of_examples = X.shape[0]

		self.number_of_classes = np.unique(Y).size

		if self.mc_tm == None:
			self.number_of_classes = np.unique(Y).size

			if self.append_negated:
				self.number_of_features = X.shape[1]*2
			else:
				self.number_of_features = X.shape[1]

			self.number_of_patches = 1
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
			self.mc_tm = _lib.CreateMultiClassTsetlinMachine(self.number_of_classes, self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)
		elif incremental == False:
			_lib.mc_tm_destroy(self.mc_tm)
			self.mc_tm = _lib.CreateMultiClassTsetlinMachine(self.number_of_classes, self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)

		if self.indexed:
			if self.itm != None:
				_lib.itm_destroy(self.itm)
			self.itm = _lib.CreateIndexedTsetlinMachine(self.mc_tm)

		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		Ym = np.ascontiguousarray(Y).astype(np.uint32)

		if self.append_negated:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
		else:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features, 1, 1, self.number_of_features, 1, 0)
		
		if self.indexed:
			_lib.itm_fit(self.itm, self.encoded_X, Ym, number_of_examples, epochs)
		else:
			_lib.mc_tm_fit(self.mc_tm, self.encoded_X, Ym, number_of_examples, epochs)

		return

	def predict(self, X):
		number_of_examples = X.shape[0]
		
		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)

		if self.append_negated:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
		else:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features, 1, 1, self.number_of_features, 1, 0)
	
		Y = np.ascontiguousarray(np.zeros(number_of_examples, dtype=np.uint32))

		if self.indexed:
			_lib.itm_predict(self.itm, self.encoded_X, Y, number_of_examples)
		else:
			_lib.mc_tm_predict(self.mc_tm, self.encoded_X, Y, number_of_examples)

		return Y
	
	def ta_state(self, mc_tm_class, clause, ta):
		return _lib.mc_tm_ta_state(self.mc_tm, mc_tm_class, clause, ta)

	def ta_action(self, mc_tm_class, clause, ta):
		return _lib.mc_tm_ta_action(self.mc_tm, mc_tm_class, clause, ta)

	def get_state(self):
		state_list = []
		for i in range(self.number_of_classes):
			ta_states = np.ascontiguousarray(np.empty(self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits, dtype=np.uint32))
			_lib.mc_tm_get_state(self.mc_tm, i, ta_states)
			state_list.append(ta_states)

		return state_list

	def set_state(self, ta_states):
		for i in range(self.number_of_classes):
			_lib.mc_tm_set_state(self.mc_tm, i, ta_states[i])

		return

	def transform(self, X, inverted=True):
		number_of_examples = X.shape[0]

		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))
		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)

		if self.append_negated:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
		else:
			_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features, 1, 1, self.number_of_features, 1, 0)
	
		X_transformed = np.ascontiguousarray(np.empty(number_of_examples*self.number_of_classes*self.number_of_clauses, dtype=np.uint32))

		if self.indexed:
			_lib.itm_transform(self.itm, self.encoded_X, X_transformed, inverted, number_of_examples)
		else:
			_lib.mc_tm_transform(self.mc_tm, self.encoded_X, X_transformed, inverted, number_of_examples)

		return X_transformed.reshape((number_of_examples, self.number_of_classes*self.number_of_clauses))

class RegressionTsetlinMachine():
	def __init__(self, number_of_clauses, T, s, boost_true_positive_feedback=1, number_of_state_bits=8, weighted_clauses=False, s_range=False):
		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)/32 + 1
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.rtm = None
		self.weighted_clauses = weighted_clauses
		if s_range:
			self.s_range = s_range
		else:
			self.s_range = s

	def __del__(self):
		if self.rtm != None:
			_lib.tm_destroy(self.rtm)

	def fit(self, X, Y, epochs=100, incremental=False):
		number_of_examples = X.shape[0]

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)

		if self.rtm == None:
			self.number_of_classes = np.unique(Y).size
			self.number_of_features = X.shape[1]*2
			self.number_of_patches = 1
			self.number_of_ta_chunks = int((self.number_of_features-1)/32 + 1)
			self.rtm = _lib.CreateTsetlinMachine(self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)
		elif incremental == False:
			_lib.tm_destroy(self.rtm)
			self.rtm = _lib.CreateTsetlinMachine(self.number_of_clauses, self.number_of_features, 1, self.number_of_ta_chunks, self.number_of_state_bits, self.T, self.s, self.s_range, self.boost_true_positive_feedback, self.weighted_clauses)

		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		Ym = np.ascontiguousarray((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)

		_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
		
		_lib.tm_fit_regression(self.rtm, self.encoded_X, Ym, number_of_examples, epochs)

		return

	def predict(self, X):
		number_of_examples = X.shape[0]
		
		self.encoded_X = np.ascontiguousarray(np.empty(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks), dtype=np.uint32))

		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		_lib.tm_encode(Xm, self.encoded_X, number_of_examples, self.number_of_features//2, 1, 1, self.number_of_features//2, 1, 1)
	
		Y = np.zeros(number_of_examples, dtype=np.int32)

		_lib.tm_predict_regression(self.rtm, self.encoded_X, Y, number_of_examples)

		return 1.0*(Y)*(self.max_y - self.min_y)/(self.T) + self.min_y
