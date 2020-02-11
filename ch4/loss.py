import numpy as py
import sys, os
sys.path.append(os.pardir)
from common.functions import *

t = [0, 0, 1, 0,0,0,0,0,0,0]

y = [0.1, 0.05, 0.9, 0, 0.05, 0, 0, 0,0,0]
l = mean_squared_error(np.array(y), np.array(t))
l = cross_entropy_error(np.array(y), np.array(t))

print(l)