import sys, os
import pickle
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.functions import *
from PIL import Image

def get_data():
	# (훈련이미지, 훈련레이블), (시험이미지, 시험레이블)
	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test


def init_network():
	with open("sample_weight.pkl", 'rb') as f:
		network = pickle.load(f)

	return network

def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)

	return y

x, t = get_data() # 시험 이미지, 시험 레이블
network = init_network()

print(x.shape)
print(x[0].shape)

accuracy_cnt = 0

for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y) # 확률이 가장 높은 원소의 index
	if p == t[i]: # label이랑 확률 제일 높은 것 같으면
		accuracy_cnt += 1

print(float(accuracy_cnt) / len(x))


###### 배치 ######
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size): # batch_size 간격으로 증가
	x_batch = x[i:i+batch_size]
	y_batch = predict(network, x_batch)
	p = np.argmax(y_batch, axis=1) # 1번째 차원 구성하는 원소에서 최대값의 인덱스를 배열로 리턴
	accuracy_cnt += np.sum(p == t[i:i+batch_size]) # 레이블이랑 같은 거 개수 더해주기

print(float(accuracy_cnt) / len(x))

## 10000개 데이터 한번에 구하는거
y = predict(network, x)
p = np.argmax(y, axis=1)
accuracy_cnt = np.sum(p == t)

print(float(accuracy_cnt) / len(x))

