import numpy as np


class Relu:
    """
    Rectified Linear Unit

    Methods:
        forward(self): Returns a NumPy nd-array of Relu of argument
        backward(self): Returns the derivative of Relu w.r.t its argument
    Arguments:
        self: A NumPy nd-array
    Returns:
        NumPy nd-array of same shape as input
    """
    def forward(self):
        return np.maximum(0, self)

    def backward(self):
        self[self <= 0] = 0
        self[self > 0] = 1
        return self


class Tanh:
    """
    Hyperbolic Tangent

    Methods:
        forward(self): Returns a NumPy nd-array of Tanh of argument
        backward(self): Returns the derivative of Tanh w.r.t its argument
    Arguments:
        self: A NumPy nd-array
    Returns:
        NumPy nd-array of same shape as input
    """
    def forward(self):
        return np.tanh(self)

    def backward(self):
        return (1 - np.square(Tanh.forward(self)))


class Sigmoid:
    """
    Sigmoid

    Methods:
        forward(self): Returns a NumPy nd-array of Sigmoid of argument
        backward(self): Returns the derivative of Sigmoid w.r.t its argument
    Arguments:
        self: A NumPy nd-array
    Returns:
        NumPy nd-array of same shape as input
    """
    def forward(self):
        return (1 / (1 + np.exp(-self)))

    def backward(self):
        return (Sigmoid.forward(self) * (1 - Sigmoid.forward(self)))


class Softmax:
    """
    Softmax

    Methods:
         forward(self): Returns a NumPy nd-array of Softmax of argument
         backward(self): Returns the derivative of Softmax w.r.t its argument
    Arguments:
        self: A NumPy nd-array
    Returns:
        NumPy nd-array of same shape as input
    """
    def forward(self):
        soft = np.exp(self) / np.sum(np.exp(self), axis=0)
        return soft

    def backward(self):
        return (Softmax.forward(self) * (1 - Softmax.forward(self)))


class Sine:
    """
    Sinusoidal

    Methods:
        forward(self): Returns a NumPy nd-array of Sine of argument
        backward(self): Returns the derivative of Sine w.r.t its argument
    Arguments:
        self: A NumPy nd-array
    Returns:
        NumPy nd-array of same shape as input
    """
    def forward(self):
        return np.sin(self)

    def backward(self):
        return np.cos(self)
    
class SoftSign:
        """
        Forward propagation
        Returns
            The output of the softplus function applied to the activation.
        
        Backward propagation
        Returns
            The derivative of SoftSign function.
        """
    def forward(self):
        return self / (np.abs(self) + 1)
    def backward(self):
        return 1/(np.square(1 + np.abs(self)))
