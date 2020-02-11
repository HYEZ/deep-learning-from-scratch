import numpy as np
from sigmoid import *

# input layer -> layer 1
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)

# layer 1 -> layer 2
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)

# layer 2 -> output layer (활성화함수만 다름)
def identify_function(x):
	return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identify_function(A3) 
# 보통 출력층 활성화함수로 리그레션=>항등함수/2-classification=>시그모이드/다중classification=>softmax
# 출력층만 다른 층들이랑 활성화 함수가 다르다.
print(Y)




