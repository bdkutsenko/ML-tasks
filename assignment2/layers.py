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
    loss=reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    # DONE: Copy from the previous assignment

    return loss, grad


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
    # DONE: Copy from the previous assignment
    # DONE implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops

    target = np.zeros(predictions.shape)
    target_index_list=target_index


    if isinstance(target_index_list, (int,np.int)):
      target[target_index]=1

    else:
      if not isinstance(target_index_list[0], (list,np.ndarray)):
        target_index_list=list(map(lambda el:[el], target_index))
      target[np.arange(predictions.shape[0]).repeat([*map(len,target_index_list)]), np.concatenate(target_index_list)]=1
    
    probs = np.zeros(predictions.shape)

    if len(predictions.shape)!=1:
      pred = predictions-np.max(predictions,axis=(len(predictions.shape)-1))[:,None]
      probs = np.exp(pred)/np.sum(np.exp(pred),axis=(len(pred.shape)-1))[:,None]
    else:
      pred = predictions-np.max(predictions)
      probs = np.exp(pred) / np.sum(np.exp(pred))

    target = np.zeros(probs.shape)
    target_index_list=target_index

    if isinstance(target_index_list,np.int) :
      target[target_index]=1
    else :
      if not isinstance(target_index_list[0], (list,np.ndarray)):
        target_index_list=list(map(lambda el:[el], target_index))
      target[np.arange(probs.shape[0]).repeat([*map(len,target_index_list)]), np.concatenate(target_index_list)]=1
    
    loss=-np.sum(target*np.log(probs))  
    dprediction=probs-target
    
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
      self.F = 0

    

    def forward(self, X):
        # DONE: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
      def ReLU(x):
        return x*(x>0)

      mapX=np.vectorize(ReLU)
      self.F = mapX(X)

      return self.F

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
        # DONE: Implement backward pass
        # Your final implementation shouldn't have any loops
        def ReLU_prime(x):
          return 1 if x > 0 else 0
        
        mapOut=np.vectorize(ReLU_prime)
        d_result=mapOut(self.F)*d_out

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # DONE: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X=X
        M=np.matmul(X,self.W.value)
        return M+np.broadcast_to(self.B.value,(M.shape))

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
        # DONE: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        d_result=np.matmul(d_out,self.W.value.T)

        self.W.grad=np.matmul(self.X.T, d_out)
        self.B.grad=np.matmul(np.ones((1,d_out.shape[0])),d_out)
        
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
