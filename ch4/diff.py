import numpy as np
import matplotlib.pyplot as plt
# 수치 미분
def numerical_diff(f, x):
	h = 1e-4
	# return (f(x+h) - f(x)) / h # 오차발생(h->0이 불가능해서 진짜 미분과 일치X)
	return (f(x+h) - f(x-h)) / (2*h) #중앙차분으로 계산

def function_1(x):
	return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

print(numerical_diff(function_1, 5))

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

# 접선을 구하는 함수
def tangent_line(f, x):
        d = numerical_diff(f, x)
        # print(d)
        y = f(x) - d*x
        return lambda t: d*t + y

# print(tangent_line(function_1, 5))
tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y2)
# plt.show()


##### 편미분 #####
def function_2(x):
	# return x[0]**2 + x[1]**2
	return np.sum(x**2)

def function_tmp1(x0):
	return x0**2 + 4.0**2

def function_tmp2(x1):
	return 3**2 + x1**2

print(numerical_diff(function_tmp2, 4))