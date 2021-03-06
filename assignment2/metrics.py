import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # DONE: Implement computing accuracy
    accuracy=np.sum(ground_truth==prediction)/ground_truth.size
    
    return accuracy
