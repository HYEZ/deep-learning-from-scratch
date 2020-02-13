## 합성곱 신경망(Convolutional Neural Network)
- `합성곱 계층`과 `폴층 계층`
	- CNN 계층은 Conv-ReLU-(Pooling)
	- 출력에 가까운 층은 지금까지의 Affine-ReLU 구성
	- 마지막 출력 계층은 Affine-Softmax로 구성
- 각 계층 사이에 3차원 데이터같이 입체적인 데이터가 흐른다는 점에서 완전연결(Fully-connected) 신경망과 다름
- 패딩(Padding), 스트라이드(Stride)와 같은 CNN 고유의 용어 등장

### 완전연결(Fully-connected) 계층의 문제점
- 데이터의 형상이 무시됨
	- ex) 데이터가 이미지인 경우 이미지는 가로, 세로, 채널로 구성된 3차원, 하지만 `fully-connected layer`에 입력 시 1차원으로 flatten 해주어야함
	- 하지만 이미지인 경우 이 형상에는 소중한 정보가 담겨있음
		- ex) 공간적으로 가까운 픽셀은 값이 비슷하거나, RGB의 각 채널은 서로 밀접하게 관련됨
		- 3차원 속에서 의미를 갖는 본질적인 패턴이 숨어있음
	- `Fully-connected layer`는 이 성질을 무시하고 `flatten`함

### 합성곱 계층(Conv)
- 데이터의 형상을 유지함
	- ex) 이미지도 3차원 데이터로 입력받으며, 다음 계층에도 3차원으로 전달함
- 따라서 CNN은 이미지처럼 형상을 가진 데이터를 제대로 이해가능함
- `특징 맵(feature map)`: 합성곱 계층의 입출력 데이터
	- `입력 특징 맵(input feature map)`: 입력 데이터
	- `출력 특징 맵(output feature map)`: 출력 데이터 

### 합성곱 연산