import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        image_channels = input_shape[2]
        
        self.layers = []
        self.layers.append(ConvolutionalLayer(image_channels, conv1_channels, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=4, stride=4))
        self.layers.append(ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1))
        self.layers.append(ReLULayer())
        self.layers.append(MaxPoolingLayer(pool_size=4, stride=4))
        self.layers.append(Flattener())
        self.layers.append(FullyConnectedLayer(4 * conv2_channels, n_output_classes))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        
        res = X.copy()
        
        for layer in self.layers:
            res = layer.forward(res)
   
        loss, dres = softmax_with_cross_entropy(res, y)
   
        for layer in reversed(self.layers):
            dres = layer.backward(dres)
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        res = X.copy()
        
        for layer in self.layers:
            res = layer.forward(res)
        
        y_pred = res.argmax(axis=1)
        return y_pred

    def params(self):
        result = {}

        for layer_num, layer in enumerate(self.layers):    
            for param_name, param in layer.params().items():
                result[param_name + '_' + str(layer_num)] = param
                
        return result
