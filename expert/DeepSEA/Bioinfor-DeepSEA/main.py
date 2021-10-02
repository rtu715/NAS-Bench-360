# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tqdm import tqdm

from model import DeepSEA
from loader import get_train_data, get_valid_data, get_test_data
from utils import plot_loss_curve, plot_roc_curve, plot_pr_curve
from utils import calculate_auroc, calculate_aupr, calculate_stats
from utils import create_dirs, write2txt, write2csv

import keras.backend as K
def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
				run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops #Prints the "flops" of the model.

np.random.seed(2)
tf.random.set_random_seed(2)

def train():
    train_dataset = get_train_data(64)
    valid_data = get_valid_data()

    # Build the model.
    model = DeepSEA()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy())
    model.build(input_shape = (None, 1000, 4))
    model.summary()
    print(get_flops())
    
    # Define the callbacks. (check_pointer\early_stopper\tensor_boarder)
    # For check_pointer: we save the model in SavedModel format
    # (Weights-only saving that contains model weights and optimizer status)
    check_pointer = tf.keras.callbacks.ModelCheckpoint(
        filepath='./result/model/ckpt',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        #save_freq='epoch',
        #load_weights_on_restart=False)
        )
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0)
    tensor_boarder = tf.keras.callbacks.TensorBoard(
        log_dir='./result/logs')

    # Training the model.
    history = model.fit(
        train_dataset,
        epochs=60,
        steps_per_epoch=71753/64,
        verbose=2,
        validation_data = valid_data,
        validation_steps=2490/64,
        callbacks=[check_pointer, early_stopper, tensor_boarder])

    # Plot the loss curve of training and validation, and save the loss value of training and validation.
    print('\n history dict: ', history.history)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_loss_curve(train_loss, val_loss, './result/model_loss.jpg')
    np.savez('./result/model_loss.npz', train_loss = train_loss, val_loss = val_loss)


def test():
    test_data = get_test_data()
    x = test_data[0]
    y = test_data[1]

    # Recreate the model.
    model = DeepSEA()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy())
    model.build(input_shape = (None, 1000, 4))
    model.summary()

    # Load the weights of the old model. (The weights content the weights of model and status of optimizer.)
    # Because the tensorflow delay the creation of variables in model and optimizer, so the optimizer status will
    # be restored when the model is trained first. like: model.train_on_batch(x[0:1], y[0:1])
    model.load_weights('./result/model/ckpt')
    # model.load_weights('./result/model/bestmodel.h5')

    result = model.predict(x) # shape = (149400, 36)

    np.savez('./result/test_result.npz',
             result = result, label = y)

    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in tqdm(range(result_shape[1]), ascii=True):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    #plot_roc_curve(fpr_list, tpr_list, './result/')
    #plot_pr_curve(precision_list, recall_list, './result/')

    header = np.array([['auroc', 'aupr']])
    content = np.stack((auroc_list, aupr_list), axis=1)
    content = np.concatenate((header, content), axis=0)
    write2csv(content, './result/result.csv')
    write2txt(content, './result/result.txt')
    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    print('AVG-AUROC:{:.3f}, AVG-AUPR:{:.3f}.\n'.format(avg_auroc, avg_aupr))
    
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def calculate_score():
    outfile = np.load('./result/test_result.npz')
    result = outfile['result']
    y = outfile['label']
    result_sig = sigmoid(result)
    stats = calculate_stats(result_sig, y)
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    print(mAP, mAUC)
    return 

if __name__ == '__main__':
    # Parses the command line arguments and returns as a simple namespace.
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.')
    args = parser.parse_args()

    # Selecting the execution mode (keras).
    create_dirs(['./result', './result/model'])
    if args.exe_mode == 'train':
        train()
    elif args.exe_mode == 'test':
        test()
        calculate_score()
