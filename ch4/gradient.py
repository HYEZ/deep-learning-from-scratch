import numpy as np
import matplotlib.pyplot as plt

def function_2(x):
	# return x[0]**2 + x[1]**2
	return np.sum(x**2)

def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

	for idx in range(x.size):
		tmp_val = x[idx]

		# f(x+h) 계산
		x[idx] = tmp_val + h
		fxh1 = f(x)

		# f(x-h) 계산
		x[idx] = tmp_val - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val

	return grad

grad = numerical_gradient(function_2, np.array([3.0, 4.0]))
print(grad)

# 경사하강법
def gradient_descent(f, init_x, lr=0.01, step_num=100):
	x = init_x

	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x -= lr * grad # 미분값만큼 x 줄여나감, lr은 학습률(learning rate)

	return x

init_x = np.array([-3.0, 4.0])
a = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
print(a)




