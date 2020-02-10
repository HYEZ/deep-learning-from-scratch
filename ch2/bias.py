import numpy as np

# theta = -b (bias)
# theta가 아닌 0으로 비교하기 위해서

def AND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = b + np.sum(x*w)

	if tmp <= 0:
		return 0 
	else:
		return 1

def NAND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5])
	b = 0.7
	tmp = b + np.sum(x*w)

	if tmp <= 0:
		return 0 
	else:
		return 1

def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.1
	tmp = b + np.sum(x*w)

	if tmp <= 0:
		return 0 
	else:
		return 1

print(AND(1, 1))