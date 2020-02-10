# 퍼셉트론 = 직선(linear)
# linear하게 하면서 xor을 표현 -> 다층 퍼셉트론 (기존 게이트 조합)
from bias import *

def XOR(x1, x2):
	s1 = NAND(x1, x2)
	s2 = OR(x1, x2)
	return AND(s1, s2)

print(XOR(1, 1))