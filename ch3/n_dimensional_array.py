import numpy as np

A = np.array([1, 2, 3, 4])
print(A)

print(np.ndim(A)) # 배열의 차수 : 1
print(A.shape) # 배열의 형상 (튜플) : 1차원 배열, 4개의 원소
# (4, ) : 다차원 배열일때랑 똑같이 나타내기 위해서 튜플로 반환 (1차원 배열, 4x1 행렬)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]]) 
print(np.ndim(B))
print(B.shape) # (3, 2) : 3x2 행렬 (all elements)

##### 행렬의 곱 (내적) #####
A = np.array([[1, 2, 3], [3, 4, 5]])
print(A.shape)
B = np.array([5, 6, 5])
print(B.shape)
print(np.dot(A, B))


# 행렬의 형상에 주의하자