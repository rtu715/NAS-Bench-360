# -*- coding: utf-8 -*-
import os
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def create_dirs(dirs):
    """
    Create dirs. (recurrent)
    :param dirs: a list directory path.
    :return: None
    """
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)

def write2txt(content, file_path):
    """
    Write array to .txt file.
    :param content: array.
    :param file_path: destination file path.
    :return: None.
    """
    try:
        file_name = file_path.split('/')[-1]
        dir_path = file_path.replace(file_name, '')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'w+') as f:
            for item in content:
                f.write(' '.join([str(i) for i in item]) + '\n')

        print("write over!")
    except IOError:
        print("fail to open file!")

def write2csv(content, file_path):
    """
    Write array to .csv file.
    :param content: array.
    :param file_path: destination file path.
    :return: None.
    """
    try:
        temp = file_path.split('/')[-1]
        temp = file_path.replace(temp, '')
        if not os.path.exists(temp):
            os.makedirs(temp)

        with open(file_path, 'w+', newline='') as f:
            csv_writer = csv.writer(f, dialect='excel')
            for item in content:
                csv_writer.writerow(item)

        print("write over!")
    except IOError:
        print("fail to open file!")

def calculate_auroc(predictions, labels):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true=labels, y_score=predictions)
    auroc = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, auroc


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


def calculate_aupr(predictions, labels):
    precision_list, recall_list, threshold_list = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
    aupr = metrics.auc(recall_list, precision_list)
    return precision_list, recall_list, aupr

def plot_loss_curve(train_loss, val_loss, file_path):
    """
    Plot the loss curve to monitor the fitting status.
    :param train_loss: (None)
    :param val_loss: same as train loss.
    :return: None
    """
    plt.figure()
    plt.plot(train_loss, lw=1, label = 'Train Loss')
    plt.plot(val_loss, lw=1, label = 'Valid Loss')
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.savefig(file_path)


def plot_roc_curve(fpr_list, tpr_list, file_path):
    """
    Plot the roc curve of 919 binary classification tasks. (DNase: 125 TFBinding: 690 Histone_Mark: 104)
    :param fpr_list: (919, None)
    :param tpr_list: (919, None)
    :param file_path: destination file path.
    :return: None
    """
    plt.figure()
    for i in range(0, 125):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"DNase I-hypersensitive sites (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_DNase.jpg'))

    plt.figure()
    for i in range(125, 815):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"Transcription factors (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_TF.jpg'))

    plt.figure()
    for i in range(815, 919):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"Histone marks (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_HistoneMark.jpg'))


def plot_pr_curve(precision_list, recall_list, file_path):
    """
    Plot the pr curve of 919 binary classification tasks. (DNase: 125 TFBinding: 690 Histone_Mark: 104)
    :param precision_list: (919, None)
    :param recall_list: (919, None)
    :param file_path: destination file path.
    :return: None.
    """
    plt.figure()
    for i in range(0, 125):
        plt.plot(precision_list[i], recall_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"DNase I-hypersensitive sites (PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_DNase.jpg'))

    plt.figure()
    for i in range(125, 815):
        plt.plot(recall_list[i], precision_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"Transcription factors (PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_TFBinding.jpg'))

    plt.figure()
    for i in range(815, 919):
        plt.plot(precision_list[i], recall_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"Histone marks (PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_HistoneMark.jpg'))
