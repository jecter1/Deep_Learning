import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
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
    max_pred_in_row = np.max(predictions, axis=1)[:, np.newaxis] if is_batch else np.max(predictions)
    pred_norm = predictions - max_pred_in_row
    pred_exp = np.exp(pred_norm)
    sum_pred_exp = np.sum(pred_exp, axis=1)[:, np.newaxis] if is_batch else np.sum(pred_exp)
    probs = pred_exp / sum_pred_exp
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
    target_probs = probs[np.arange(target_index.size), target_index] if is_batch else probs[target_index]
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
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    is_batch = probs.ndim == 2
    dprediction = probs
    
    if is_batch:
        dprediction[np.arange(target_index.size), target_index] -= 1
        dprediction /= target_index.size
    else:
        dprediction[target_index] -= 1
        
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        res = np.maximum(0, X)
        return res

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
        d_result[self.X < 0] = 0
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
        res = np.dot(X, self.W.value) + self.B.value
        return res

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
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)
        d_result = np.dot(d_out, self.W.value.T)
        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):        
        batch_size, height, width, _ = X.shape
        height_pad = height + 2 * self.padding
        width_pad = width + 2 * self.padding
        
        self.batch_size = batch_size
        self.X_pad = np.zeros((batch_size, height_pad, width_pad, self.in_channels))
        self.last_padding_y_index = height_pad - self.padding
        self.last_padding_x_index = width_pad - self.padding
        self.X_pad[:, self.padding:self.last_padding_y_index, self.padding:self.last_padding_x_index, :] = X
        
        out_height = height_pad - self.filter_size + 1
        out_width = width_pad - self.filter_size + 1
        
        flat_W = self.W.value.reshape(-1, self.out_channels)
        
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        for y in range(out_height):
            for x in range(out_width):
                last_y = y + self.filter_size
                last_x = x + self.filter_size
                current_input = self.X_pad[:, y:last_y, x:last_x, :].reshape(batch_size, -1)
                result[:, y, x, :] = np.dot(current_input, flat_W) + self.B.value
                
        return result


    def backward(self, d_out):
        d_in = np.zeros_like(self.X_pad)
        
        _, out_height, out_width, out_channels = d_out.shape
        
        for y in range(out_height):
            for x in range(out_width):
                last_y = y + self.filter_size
                last_x = x + self.filter_size
                
                current_input = self.X_pad[:, y:last_y, x:last_x, :].reshape(self.batch_size, -1)
                
                W_grad_inc_disordered = np.dot(current_input.T, d_out[:, y, x, :])
                W_grad_inc = W_grad_inc_disordered.reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)
                self.W.grad += W_grad_inc
                
                X_grad_inc_disordered = np.dot(d_out[:, y, x, :], self.W.value.reshape(-1, out_channels).T)
                X_grad_inc = X_grad_inc_disordered.reshape(self.batch_size, self.filter_size, 
                                                           self.filter_size, self.in_channels)
                d_in[:, y:last_y, x:last_x, :] += X_grad_inc
                
                
        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
                
        return d_in[:, self.padding:self.last_padding_y_index, self.padding:self.last_padding_x_index, :]
    
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        
        res_height = height // self.stride - self.pool_size + self.stride
        res_width = width // self.stride - self.pool_size + self.stride
        result = np.zeros((batch_size, res_height, res_width, channels))
        
        for y in range(res_height):
            for x in range(res_width):
                y_first = y * self.stride
                y_last = y_first + self.pool_size
                x_first = x * self.stride
                x_last = x_first + self.pool_size
                result[:, y, x, :] = np.max(X[:, y_first:y_last, x_first:x_last, :], axis=(1, 2))
        
        return result
        
    def backward(self, d_out):
        dX = np.zeros_like(self.X)
        batch_size, height, width, channels = d_out.shape
        
        for y in range(height):
            for x in range(width):
                y_first = y * self.stride
                y_last = y_first + self.pool_size
                x_first = x * self.stride
                x_last = x_first + self.pool_size
                sliced_X = self.X[:, y_first:y_last, x_first:x_last, :]
                sliced_grad = d_out[:, y:y+1, x:x+1, :]
                
                max_sliced_X = np.max(sliced_X, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
                
                is_max = sliced_X == max_sliced_X
                dX[:, y_first:y_last, x_first:x_last, :] += sliced_grad * is_max
        
        return dX

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        out = X.reshape(self.X_shape[0], -1)
        return out

    def backward(self, d_out):
        dX = d_out.reshape(self.X_shape)
        return dX
        
    def params(self):
        # No params!
        return {}
