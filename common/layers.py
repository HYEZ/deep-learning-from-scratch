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
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
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