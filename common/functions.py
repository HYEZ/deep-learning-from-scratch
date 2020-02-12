import numpy as np
from PIL import Image

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# def softmax(a):
# 	c = np.max(a)
# 	exp_a = np.exp(a-c)
# 	sum_exp_a = np.sum(exp_a)
# 	y = exp_a / sum_exp_a
	
# 	return y
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))
    
def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()

def mean_squared_error(y, t):
	return 0.5 * np.sum((y-t)**2)

# def cross_entropy_error(y, t):
# 	delta = 1e-7
# 	return -np.sum(t * np.log(y + delta)) # log0 방지

# def cross_entropy_error(y, t):
# 	delta = 1e-7
# 	if y.ndim == 1:
# 		t = t.reshape(1, t.size)
# 		y = y.reshape(1, y.size)
# 	batch_size = y.shape[0]
# 	print("batch_size", batch_size)
	# return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size