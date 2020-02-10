# 리스트
a = [1, 2, 3, 4, 5]
print(a)

# 인덱스 슬라이싱
print(a[:1])

# 딕셔너리
me = {
	"height": 100
}
print(me["height"])
me["weight"] = 30
print(me)

# for 문
for i in [1, 2, 3]:
	print(i)

# class (메서드 첫번째 파라미터에 self를 써야함)
class Man:
	def __init__(self, name):
		self.name = name
		print("초기화")

	def hello(self):
		print(self.name)

m = Man("David")
m.hello()
