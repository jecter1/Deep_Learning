import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    is_batch = predictions.ndim == 2
    
    max_in_row = np.amax(predictions, axis=1)[:, np.newaxis] if is_batch else np.max(predictions)
    exp_pred = np.exp(predictions - max_in_row)
    exp_sum = np.sum(exp_pred, axis=1)[:, np.newaxis] if is_batch else np.sum(exp_pred)
        
    probs = exp_pred / exp_sum
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    is_batch = probs.ndim == 2
    
    if is_batch:
        i = np.arange(target_index.size)
        target_probs = probs[i, target_index]
    else:
        target_probs = probs[target_index]
    loss = -np.mean(np.log(target_probs))
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    is_batch = predictions.ndim == 2
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if is_batch:
        batch_size = target_index.size
        i = np.arange(batch_size)
        dprediction[i, target_index] -= 1
        dprediction /= batch_size
    else:
        dprediction[target_index] -= 1
    
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.negative = X < 0
        result = np.maximum(0, X)
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out.copy()
        d_result[self.negative] = 0
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        d_result = np.dot(d_out, self.W.value.T)
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0).reshape(1, -1)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
