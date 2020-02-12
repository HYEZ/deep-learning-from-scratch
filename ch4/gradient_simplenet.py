import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import *

class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2, 3) # 정규분포로 초기화

	# 예측
	def predict(self, x):
		return np.dot(x, self.W) # X x W

	# 손실함수의 값을 구하는 함수
	def loss(self, x, t): 
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y ,t)
		return loss


net = simpleNet() 
x = np.array([0.6, 0.9])
p = net.predict(x)
print(np.argmax(p)) # 최대값 인덱스
print(p)
t = np.array([1, 0, 0]) # 정답 레이블
print(net.loss(x, t))

f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W) # 미분 한번 한 것! 경사하강법으로 기울기 0 되도록 점점 조정해 나갈 것.
print(dW)

