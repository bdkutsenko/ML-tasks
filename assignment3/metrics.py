import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    TP=np.sum(prediction & ground_truth)
    TN=np.sum(~(prediction + ground_truth))
    FP=np.sum(~(ground_truth+~prediction))
    FN=np.sum(~(~ground_truth+prediction))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
  
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = 2*precision*recall/(recall+precision)

    # DONE: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    accuracy=np.sum(ground_truth==prediction)/ground_truth.size
   
    return accuracy

