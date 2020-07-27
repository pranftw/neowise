from neowise.CostFunction import Cost
import numpy as np

y = np.array([1, 1, 0, 0])
A = np.array([0.65, 0.2, 0.6, 0.99])
cost = Cost.binary_cross_entropy(y=y, A=A, m=4)
print(cost)
