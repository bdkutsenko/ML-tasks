import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.l1=FullyConnectedLayer(n_input, hidden_layer_size);
        self.ReLU=ReLULayer();
        self.l2=FullyConnectedLayer(hidden_layer_size, n_output);
        
        # DONE Create necessary layers

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # DONE Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        params = self.params()

        for param_key in params:
          params[param_key].grad=0

        # DONE Compute loss and fill param gradients
        # by running forward and backward passes through the model

        predictions=self.l2.forward(self.ReLU.forward(self.l1.forward(X)))
        loss, dprediction=softmax_with_cross_entropy(predictions,y)

        self.l1.backward(self.ReLU.backward(self.l2.backward(dprediction)))

        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        for param_key in params:
          regloss,regprediction = l2_regularization(params[param_key].value,self.reg)
          loss+=regloss
          params[param_key].grad+=regprediction

        
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # DONE: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        sample_pred = self.l2.forward(self.ReLU.forward(self.l1.forward(X)))
        pred = np.argmax(sample_pred,axis=1)

        return pred

    def params(self):
        result = {'W1': self.l1.W,'B1': self.l1.B, 'W2': self.l2.W,"B2 ": self.l2.B }

        # DONE Implement aggregating all of the params

        return result
