import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    TP = np.sum(prediction & ground_truth)
    TN = np.sum(~prediction & ~ground_truth)
    FP = np.sum(prediction & ~ground_truth)
    FN = np.sum(~prediction & ground_truth)
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    true_cnt = np.sum(prediction == ground_truth)
    false_cnt = np.sum(prediction != ground_truth)
    
    return true_cnt / (true_cnt + false_cnt)
