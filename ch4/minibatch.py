import numpy as py
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from dataset.mnist import load_mnist

# (훈련이미지, 훈련레이블), (시험이미지, 시험레이블)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
# one_hot_label=True : label을 배열로 얻음 (정답부분만 1이게) 
print(x_train.shape)
print(t_test.shape)

##### 미니배치 = 샘플링 개념 #####
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(t_batch)