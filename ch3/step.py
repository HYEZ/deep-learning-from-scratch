# 단층 퍼셉트론 = 게단함수를 활성화함수로 사용하는 모델
# 다층 퍼셉트론 = 신경망(시그모이드 등)을 활성화 함수로 사용하는 모델

import numpy as np
import matplotlib.pyplot as plt

def ori_step_function(x):
	if x > 0:
		return 1
	else: 
		return 0

def step_function(x):
	y = x > 0
	return y.astype(np.int) # astype: numpy 배열 형변환 (bool => int로 변환)

x = np.array([1.0, 2.0])
print(x)
print(step_function(x))


x = np.arange(-5.0, 5.0, 0.1) # -5 부터 5 전까지 0.1 단위로 배열 생성
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()

