from neowise.ActivationFunction import ActivationsForward
import numpy as np
relu_back = ActivationsForward.relu(np.array([0,-0.5,1,2]))
print(relu_back)