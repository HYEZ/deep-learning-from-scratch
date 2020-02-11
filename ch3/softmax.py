import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
print(y)

def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a-c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	
	return y

##### softmax 구현 시 오버플로우 조심하기!!!

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(np.sum(y)) 
# softmax 출력의 총합은 항상 1이다 => 확률로 해석 => 가장 높은 값(확률)을 가진걸 선택
# softmax 함수 사용해도 원소의 대소관계는 그대로 (지수함수가 단조증가함수라서) => 사실 softmax 생략해도 됨
# 보통 train은 softmax 사용, predit는 사용X


