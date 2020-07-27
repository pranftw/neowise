from neowise import ActivationFunction
import numpy as np

g = np.array([1, 2, 3, 4])
relu = ActivationFunction.ActivationsForward.softmax
f = relu(g)
print(f)
