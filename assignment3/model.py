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
        # DONE Create necessary layers
        self.cl1 = ConvolutionalLayer(3, conv1_channels, 3, 1);
        self.ReLU1 = ReLULayer();
        self.mp1 = MaxPoolingLayer(4,4);
        self.cl2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1);
        self.ReLU2 = ReLULayer();
        self.mp2 = MaxPoolingLayer(4,4);
        self.flat = Flattener();
        self.l3 = FullyConnectedLayer(4*conv2_channels,n_output_classes);
        




    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        
        params = self.params()
        for param_key in params:
          params[param_key].grad=0

        # DONE Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        self.predictions=self.l3.forward(self.flat.forward(self.mp2.forward(self.ReLU2.forward(self.cl2.forward(self.mp1.forward(self.ReLU1.forward(self.cl1.forward(X))))))))
        loss, dprediction=softmax_with_cross_entropy(self.predictions,y)
        #print(self.mp2.backward(self.flat.backward(self.l3.backward(dprediction))))
        self.cl1.backward(self.ReLU1.backward(self.mp1.backward(self.cl2.backward(self.ReLU2.backward(self.mp2.backward(self.flat.backward(self.l3.backward(dprediction))))))))
        #print(self.cl2.W.grad)
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)
        #print(self.predictions)
        predictions = self.l3.forward(self.flat.forward(self.mp2.forward(self.ReLU2.forward(self.cl2.forward(self.mp1.forward(self.ReLU1.forward(self.cl1.forward(X))))))))
        pred = np.argmax(predictions,axis=1)
        #print(pred)
        return pred

    def params(self):
        result = {'W1': self.cl1.W,'B1': self.cl1.B, 'W2': self.cl2.W,"B2 ": self.cl2.B,'W3': self.l3.W,"B3 ": self.l3.B }

        # DONE: Aggregate all the params from all the layers
        # which have parameters

        return result
