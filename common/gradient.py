import numpy as np

def numerical_gradient(f, x):
	h = 1e-4
	ori_x = x
	grad = np.zeros_like(x) # x와 형상이 같은 배열 생성
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index # 배열의 인덱스를 튜플로 반환
		print(idx)
		tmp_val = x[idx]

		# f(x+h) 계산
		x[idx] = float(tmp_val) + h
		fxh1 = f(x)

		# f(x-h) 계산
		x[idx] = float(tmp_val) - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val

		it.iternext()

	return grad