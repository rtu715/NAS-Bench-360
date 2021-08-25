import os
import numpy as np
from sklearn import metrics

def load_deepsea_data(path, train):
    data = np.load(os.path.join(path, 'deepsea_filtered.npz'))
    train_data, train_labels = data['x_train'], data['y_train']

    val_data, val_labels = data['x_val'], data['y_val']

    if train: 
        return train_data, train_labels, val_data, val_labels

    test_data, test_labels = data['x_test'], data['y_test']

    return (train_data, train_labels), (test_data, test_labels)

def calculate_stats(output, target, class_indices=None):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      class_indices: list
        explicit indices of classes to calculate statistics for

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    # Class-wise statistics
    for k in class_indices:
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)


        dict = {'AP': avg_precision,
                'auc': auc}
        stats.append(dict)

    return stats