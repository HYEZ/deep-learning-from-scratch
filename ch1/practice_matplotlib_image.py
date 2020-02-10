import matplotlib.pyplot as plt
from matplotlib.image import imread # 이렇게 import하면 .안쓰고 함수 바로 쓸 수 있음

img = imread("../data/book.jpg")

plt.imshow(img)
plt.show()