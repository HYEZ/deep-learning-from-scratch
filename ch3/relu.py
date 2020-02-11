import numpy as np
import matplotlib.pyplot as plt

def relu(x):
	return np.maximum(0, x)

# 입력이 0을 넘으면 입력 그대로 출력, 이하면 0을 출력

