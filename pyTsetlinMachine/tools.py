import numpy as np

class Binarizer:
	def __init__(self, max_bits_per_feature = 25):
		self.max_bits_per_feature = max_bits_per_feature
		return

	def fit(self, X):
		self.number_of_features = 0
		self.unique_values = []
		for i in range(X.shape[1]):
			uv = np.unique(X[:,i])[1:]
			if uv.size > self.max_bits_per_feature:
				unique_values = np.empty(0)

				step_size = 1.0*uv.size/self.max_bits_per_feature
				pos = 0.0
				while int(pos) < uv.size and unique_values.size < self.max_bits_per_feature:
					unique_values = np.append(unique_values, np.array(uv[int(pos)]))
					pos += step_size
			else:
				unique_values = uv

			self.unique_values.append(unique_values)
			self.number_of_features += self.unique_values[-1].size
		return

	def transform(self, X):
		X_transformed = np.zeros((X.shape[0], self.number_of_features))

		pos = 0
		for i in range(X.shape[1]):
			for j in range(self.unique_values[i].size):
				X_transformed[:,pos] = (X[:,i] >= self.unique_values[i][j])
				pos += 1

		return X_transformed