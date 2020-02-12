import numpy as np
import sys, os
sys.path.append(os.pardir)  
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []


# 하이퍼 파라미터
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100 # 미니배치 크기
learning_rate = 0.1
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 1에폭 횟수 : 60000/100=600 (1 epoch = 600)
inter_per_epoch = max(train_size / batch_size, 1)
print(inter_per_epoch)


for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size) # train_size 중에서 batch_size 만큼 랜덤 선택
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# 기울기 계산
	grad = network.gradient(x_batch, t_batch)

	# 매개변수 갱신
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]


	# 학습 경과 기록
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)
	# print(loss)

	# 1에폭당 정확도 계산
	# print(i, inter_per_epoch)
	if i % inter_per_epoch == 0: 
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc, test acc", str(train_acc) + "," + str(test_acc))

print(train_loss_list)


