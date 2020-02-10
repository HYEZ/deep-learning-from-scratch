import numpy as np # numpy를 np라는 이름으로 가져옴

x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 3.0])
print(x+y) # 산술연산 가능
z = x+y
print(type(z))

# 브로드캐스트
print(x / 20)

# 넘파이 N차원 배열
A = np.array([[1, 2], [2, 3]])
print(A)
print(A.shape) # 2X2 행렬의 형상
print(A.dtype) # 원소의 자료형

B = np.array([[5, 2], [3, 3]])
print(A + B) # 산술연산
print(A + 2) # 브로드캐스트

# 배열끼리 브로드캐스트
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A*B)

# 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X[0][1])
for row in X:
	print(row)

# 1차원 배열로 변환
X = X.flatten()
print(X)

print(X[np.array([0, 2, 4])]) # 인덱스가 0, 2, 4인 것만 가져오기
print(X>15)
print(X[X>15])