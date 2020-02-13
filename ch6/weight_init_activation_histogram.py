import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100) # 1000개의 데이터
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
	if i != 0:
		x = activations[i-1]

	w = np.random.randn(node_num, node_num) / np.sqrt(node_num) 
	a = np.dot(x, w)
	z = sigmoid(a)
	activations[i] = z

for i, a in activations.items():
	print("i: ", i)
	print("a: ", a.flatten().shape, a.shape)
	plt.subplot(1, len(activations), i+1) # nrows, ncols, index
	plt.title(str(i+1) + "-layer")
	plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()

