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
    loss=reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    # DONE: Copy from the previous assignment

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
      self.F = 0
      

    def forward(self, X):
        # DONE copy from the previous assignment
        # DONE: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
      def ReLU(x):
        return x*(x>0)

      mapX=np.vectorize(ReLU)
      self.F = mapX(X)

      return self.F

    def backward(self, d_out):
        # DONE: Implement backward pass
        # Your final implementation shouldn't have any loops
        def ReLU_prime(x):
          return 1 if x > 0 else 0
        mapOut=np.vectorize(ReLU_prime)
        d_result=mapOut(self.F)*d_out

        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # DONE copy from the previous assignment
        self.X=X
        M=np.matmul(X,self.W.value)
        return M+np.broadcast_to(self.B.value,(M.shape))

    def backward(self, d_out):
      # DONE copy from the previous assignment
        d_input=np.matmul(d_out,self.W.value.T)

        self.W.grad=np.matmul(self.X.T, d_out)
        self.B.grad=np.matmul(np.ones((1,d_out.shape[0])),d_out)
            
        return d_input

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
        batch_size, height, width, channels = X.shape

        out_height = height - (self.filter_size-1)+2*self.padding 
        out_width =  width - (self.filter_size-1)+2*self.padding 
        
        # DONE: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        #print(np.pad(X,((0, 0), (self.padding, self.padding), (self.padding, self.padding),(0,0)),'constant',constant_values=0)[0,1:1+self.filter_size,1:1+self.filter_size,0])

        self.X=np.pad(X,((0, 0), (self.padding, self.padding), (self.padding, self.padding),(0,0)),'constant',constant_values=0)
        #self.X = X
        d_results=np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                # DONE: Implement forward pass for specific location
                #print(np.pad(X,((0, 0), (self.padding, self.padding), (self.padding, self.padding),(0,0)),'constant',constant_values=0)[:,y:y+self.filter_size,x:x+self.filter_size,:].shape)
                d_results[:,y,x,:] = np.matmul(np.pad(X,((0, 0), (self.padding, self.padding), (self.padding, self.padding),(0,0)),'constant',constant_values=0)[:,y:(y+self.filter_size),x:(x+self.filter_size),:].reshape((batch_size,self.filter_size*self.filter_size*self.in_channels)),self.W.value.reshape((self.filter_size*self.filter_size*self.in_channels,self.out_channels)))
        return d_results+np.broadcast_to(self.B.value,(d_results.shape))


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # DONE: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        # Try to avoid having any other loops here too
        d_results=np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                # DONE: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B) 
                #print(np.pad(np.matmul(d_out[:,y,x,:].reshape(batch_size,self.out_channels),self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels,self.out_channels).T).reshape(batch_size,self.filter_size,self.filter_size,self.in_channels),((0, 0), (0, self.padding), (0, self.padding),(0,0)),'constant',constant_values=0))
                #print(d_results[:,:,:,:].shape)

                d_results[:,y:(y+self.filter_size),x:(x+self.filter_size),:] += np.matmul(d_out[:,y,x,:].reshape((batch_size,self.out_channels)),self.W.value.reshape((self.filter_size*self.filter_size*self.in_channels,self.out_channels)).T).reshape((batch_size,self.filter_size,self.filter_size,self.in_channels))
                self.W.grad += np.matmul(self.X[:,y:(y+self.filter_size),x:(x+self.filter_size),:].reshape(batch_size,self.filter_size*self.filter_size*self.in_channels).T,d_out[:,y,x,:].reshape((batch_size,self.out_channels))).reshape((self.filter_size,self.filter_size,self.in_channels,self.out_channels))
                #self.B.grad += np.matmul(np.ones_like(self.B.value),d_out[:,y,x,:].reshape(batch_size,self.out_channels))
        self.B.grad=np.reshape(np.sum(d_out,axis=(0,1,2)),self.B.value.shape)

        if(self.padding==0):
          return d_results
        else:
          return d_results[:,self.padding:-self.padding,self.padding:-self.padding,:]
        

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

        # DONE: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = np.int((height + self.pool_size-1)/self.stride) 
        out_width =  np.int((width + self.pool_size-1)/self.stride)


        d_results=np.zeros((batch_size,out_height,out_width,channels))

        self.X=X

        for y in range(out_height):
            for x in range(out_width):           
              d_results[:,y,x,:] = np.amax(X[:,self.stride*y:(self.stride*y+self.pool_size),self.stride*x:(self.stride*x+self.pool_size),:],axis=(1,2))



      
        return d_results

    def backward(self, d_out):
        # DONE: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        #d_results=np.zeros((batch_size,height,width,channels))
        d_results=np.zeros(self.X.shape)


        for y in range(out_height):
            for x in range(out_width):
              for i in range(batch_size):
                for j in range(channels):
                  X = self.X[i,self.stride*y:(self.stride*y+self.pool_size),self.stride*x:(self.stride*x+self.pool_size),j]
                  index_tmp = np.unravel_index(np.argmax(X),(X.shape[0],X.shape[1]))
                  #index_tmp = np.unravel_index(np.argwhere(self.X[i,self.stride*y:(self.stride*y+self.pool_size),self.stride*x:(self.stride*x+self.pool_size),j] == d_results[i,y,x,j]),(height,width))
                  #print(d_out[i,y,x,j])
                  #if((d_out[i,y,x,j] == self.X[:,self.stride*y:(self.stride*y+self.pool_size),self.stride*x:(self.stride*x+self.pool_size),:]).any()):
                   #d_results[i,y,x,j] = d_out[i,y,x,j]
                  d_results[i,self.stride*y+index_tmp[0],self.stride*x+index_tmp[1],j] = d_out[i,y,x,j]
                  

        return d_results

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # DONE: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]     
        self.X = X

        return X.reshape((batch_size, height*width*channels))

    def backward(self, d_out):
        # DONE: Implement backward pass
        return d_out.reshape(self.X.shape)

    def params(self):
        # No params!
        return {}
