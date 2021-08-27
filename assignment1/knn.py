import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
               dists[i_test][i_train]=np.sum(np.abs(self.train_X[i_train]-X[i_test]))
                # DONE: Fill dists[i_test][i_train]
      
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
   
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        #print(np.sum(np.abs(self.train_X-np.tile(X[0],(num_train,1))),axis=1).shape)
      
        
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
           #dists[i_test]=np.sum(np.abs(self.train_X-np.tile(X[i_test],(num_train,1))),axis=1)
           dists[i_test]=np.sum(np.abs(self.train_X-np.broadcast_to(X[i_test, :,np.newaxis],(3072 ,num_train)).transpose(1, 0)),axis=1)
            # DONE: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)

        #dists=np.sum(np.abs(np.repeat(self.train_X[:, :, np.newaxis], num_test, axis=2).transpose(2, 1, 0)-np.repeat(X[:, :, np.newaxis], num_train, axis=2)),axis=1)
        dists=np.sum(np.abs(np.broadcast_to(self.train_X[:, :, np.newaxis],(num_train,3072,num_test)).transpose(2, 1, 0)-np.broadcast_to(X[:, :, np.newaxis],(num_test,3072,num_train))),axis=1)
        # DONE: Implement computing all distances with no loops!
        
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        idx=np.argpartition(dists[0], self.k)
        #print(np.sum(self.train_y[idx[:self.k]]))
        #print(self.train_y[idx[:self.k]].size)

        for i in range(num_test):
          idx=np.argpartition(dists[i], self.k)
          if np.sum(self.train_y[idx[:self.k]])>self.train_y[idx[:self.k]].size/2:
            pred[i]=True
          elif np.sum(self.train_y[idx[:self.k]])<self.train_y[idx[:self.k]].size/2:
            pred[i]=False
          else:
            pred[i]=self.train_y[idx[:1]]
            # DONE: Implement choosing best class based on k
            # nearest training samples
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
       
        for i in range(num_test):
          idx=np.argpartition(dists[i], self.k)
          #print(self.train_y.shape)
          digit= np.zeros(10, np.int)
          for j in range (self.k):
            digit[self.train_y[idx[j]]]+=1
          pred[i] = np.argmax(digit)
            # DONE: Implement choosing best class based on k

        return pred
