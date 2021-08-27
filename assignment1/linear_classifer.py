import numpy as np


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
    # DONE implement softmax
    # Your final implementation shouldn't have any loops

    
    probs = np.zeros(predictions.shape)
    if len(predictions.shape)!=1:
      pred = predictions-np.max(predictions,axis=(len(predictions.shape)-1))[:,None]
      probs = np.exp(pred)/np.sum(np.exp(pred),axis=(len(pred.shape)-1))[:,None]
    else:
      pred = predictions-np.max(predictions)
      probs = np.exp(pred) / np.sum(np.exp(pred))

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
    # DONE implement cross-entropy
    # Your final implementation shouldn't have any loops

    
    target = np.zeros(probs.shape)
    target_index_list=target_index

    if isinstance(target_index_list,np.int) :
      target[target_index]=1
    else :
      if not isinstance(target_index_list[0], (list,np.ndarray)):
        target_index_list=list(map(lambda el:[el], target_index))
      target[np.arange(probs.shape[0]).repeat([*map(len,target_index_list)]), np.concatenate(target_index_list)]=1
    
    loss=-np.sum(target*np.log(probs))  
    


    return loss


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
    
    probs=softmax(predictions)
    loss=cross_entropy_loss(probs, target_index)
    dprediction=(probs-target)
    #print(probs," - ",target," = ", dprediction)
    
    return loss, dprediction


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


    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss=reg_strength*np.sum(np.multiply(W,W))
    grad = 2*reg_strength*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    
    # DONE implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    dW=np.zeros(W.shape)
    predictions = np.dot(X, W)
    loss,dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW=np.matmul(X.T, dprediction)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1

        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
          for i in range(int(X.shape[0]/batch_size)):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            # DONE implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            #print(batches_indices[0])

            batch = X[batches_indices[i],:]
            target_index = list(y[batches_indices[i],None])
            #print(i)
            #print(len(batches_indices[0]))
            #print(batch.shape)
            
            soft_loss, dW = linear_softmax(batch, self.W, target_index)
            reg_loss, reg_grad = l2_regularization(self.W, reg)
            loss = soft_loss+reg_loss

            #print(reg_loss)
            self.W-=learning_rate*(dW+reg_grad)
            #print(self.W[0][0])
            #print(reg_grad.shape, dW.shape, self.W.shape )
            #print(batch[0][0][0])
            
            # end
          if epoch % 50 == 0:
            print("Epoch %i, loss: %f" % (epoch, loss))
          loss_history.append(loss)
           
        print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # DONE Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions=np.matmul(X,self.W)
        y_pred=np.argmax(predictions,axis=1)
        #print(y_pred.shape)

        return y_pred



                
                                                          

            

                
