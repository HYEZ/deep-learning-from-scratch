import numpy as np
from common.functions import *

class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0) # x가 0보다 작으면 Ture
		out = x.copy() # numpy는 대입할 경우 복사X, 포인터 복사같이 주소 복사하는 개념으로 보면 될듯. copy써야 데이터만 복사함
		out[self.mask] = 0
		return out

	def backward(self, dout):
		# print(dout[self.mask])
		
		# print(self.mask.shape)
		# dout[self.mask] = 0
		dx = dout

		return dx

# 시그모이드의 역전파는 순전파의 출력 y만으로 계산 가능하다. 
class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out

		return out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out

		return dx

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None
		self.original_x_shape = None

	def forward(self, x):
		self.x = x
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)

		out = np.dot(x, self.W) + self.b


		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)
		dx = dx.reshape(*self.original_x_shape)

		return dx

class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None # loss
		self.y = None # softmax 출력
		self.t = None # 정답레이블 (one-hot vector)

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)
		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		# print(self.y)
		# print(self.t)
		# print( self.y-self.t)
		# print(batch_size)
		dx = (self.y - self.t) / batch_size

		return dx

# test = np.array([[1, 2], [3, 4]])
# b = np.array([1, 2])
# print(test+b)

# dy = np.array([[1, 2, 3], [4, 5, 6]])
# db = np.sum(dy, axis=0)
# print(db)

# test = SoftmaxWithLoss()
# a = test.forward(np.array([[1, -2], [2, -1]]), np.array([1, 0]))
# b = test.backward()
# print(b)

# test = Sigmoid()
# a = test.forward(np.array([[1, -2],[-1, 2]]))
# b = test.backward(np.array([[1, -3],[-1, 2]]))
# print(b)