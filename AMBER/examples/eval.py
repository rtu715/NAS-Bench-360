import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils.np_utils import to_categorical   
from sklearn.metrics import precision_score, recall_score

import amber
import os
import shutil
from amber.modeler import KerasResidualCnnBuilder
from amber.architect import ModelSpace, Operation
from amber.utils.io import read_history

from load_ecg import read_data_physionet_4, custom_f1, f1_score
from load_satellite import load_satellite_data 


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_recall': recall_score(y_val, y_predict),
            'val_precision': precision_score(y_val, y_predict),
        })
        return

    def get_data(self):
        return self._data


def get_model_space(out_filters=64, num_layers=9):
    model_space = ModelSpace()
    num_pool = 4
    expand_layers = [num_layers//4-1, num_layers//4*2-1, num_layers//4*3-1]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu', dilation=10),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu', dilation=10),
            # max/avg pool has underlying 1x1 conv
            Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space

def main():
    args = sys.argv[1:]
    metrics = Metrics()

    wd = '.'
    if args[0] == 'ECG':
        input_node = Operation('input', shape=(1000, 1), name="input")
        output_node = Operation('dense', units=4, activation='sigmoid')
        
        X_train, Y_train, X_test, Y_test, pid_test = read_data_physionet_4('.')
        Y_train = to_categorical(Y_train, num_classes=4)
        Y_test = to_categorical(Y_test, num_classes=4)
        bs = 128

    elif args[0] == 'satellite':
        input_node = Operation('input', shape=(46, 1), name="input")
        output_node = Operation('dense', units=24, activation='sigmoid')

        X_train, Y_train, X_test, Y_test = load_satellite_data('.', False)
        Y_train = to_categorical(Y_train, num_classes=24)
        Y_test = to_categorical(Y_test, num_classes=24)
        bs = 1024

    else:
        raise NotImplementedError

    
    hist = read_history([os.path.join(wd, "train_history.csv")], 
                        metric_name_dict={'zero':0, 'auc': 1})
    hist = hist.sort_values(by='auc', ascending=False)
    hist.head(n=5)
    model_space = get_model_space(out_filters=32, num_layers=12)

    keras_builder = KerasResidualCnnBuilder(inputs_op=input_node,
        output_op=output_node,
        fc_units=100,
        flatten_mode='Flatten',
        model_compile_dict={
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam',
            'metrics': ['acc']
            },
        model_space=model_space,
        dropout_rate=0.1,
        wsf=2
        )

    best_arc = hist.iloc[0][[x for x in hist.columns if x.startswith('L')]].tolist()
    searched_mod = keras_builder(best_arc)
    searched_mod.summary()

    history =  searched_mod.fit(
        X_train,
        Y_train,
        batch_size=bs,
        validation_data=(X_test, Y_test),
        epochs=100,
        verbose=1,
        callbacks=[metrics]
    )

    if args[0] == 'ECG':
        all_pred_prob = []
        for item in X_test:
            logits = searched_mod.predict(item)
            all_pred_prob.append(logits.eval())

        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
        final= f1_score(all_pred, Y_test, pid_test)
        print(final)
        np.savetxt('score.txt', np.array(final))
    
    np.savetxt('metrics.txt',history.history)

