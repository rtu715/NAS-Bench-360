import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
import json
import amber
import os
import shutil
from amber.modeler import KerasResidualCnnBuilder
from amber.architect import ModelSpace, Operation
from amber.utils.io import read_history

from load_ecg import read_data_physionet_4, custom_f1, f1_score
from load_satellite import load_satellite_data
from load_deepsea import load_deepsea_data, calculate_stats


class RocCallback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' %
              (str(round(roc_train, 4)), str(round(roc_val, 4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


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


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def get_model_space(out_filters=64, num_layers=9):
    model_space = ModelSpace()
    num_pool = 4
    expand_layers = [num_layers//4-1, num_layers//4*2-1, num_layers//4*3-1]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters,
                      kernel_size=8, activation='relu'),
            Operation('conv1d', filters=out_filters,
                      kernel_size=4, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=8,
                      activation='relu', dilation=10),
            Operation('conv1d', filters=out_filters, kernel_size=4,
                      activation='relu', dilation=10),
            # max/avg pool has underlying 1x1 conv
            Operation('maxpool1d', filters=out_filters,
                      pool_size=4, strides=1),
            Operation('avgpool1d', filters=out_filters,
                      pool_size=4, strides=1),
            Operation('identity', filters=out_filters),
        ])
        if i in expand_layers:
            out_filters *= 2
    return model_space


import keras.backend as K

def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
				run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops #Prints the "flops" of the model.

def main():
    args = sys.argv[1:]

    wd = '.'
    if args[0] == 'ECG':
        input_node = Operation('input', shape=(1000, 1), name="input")
        output_node = Operation('dense', units=4, activation='softmax')

        X_train, Y_train, X_test, Y_test, pid_test = read_data_physionet_4(wd)
        Y_train = to_categorical(Y_train, num_classes=4)
        Y_test = to_categorical(Y_test, num_classes=4)
        bs = 512
        loss = 'categorical_crossentropy'

    elif args[0] == 'satellite':
        input_node = Operation('input', shape=(46, 1), name="input")
        output_node = Operation('dense', units=24, activation='softmax')

        X_train, Y_train, X_test, Y_test = load_satellite_data(wd, False)
        bs = 1024
        loss = 'categorical_crossentropy'

    elif args[0] == 'deepsea':
        input_node = Operation('input', shape=(1000, 4), name="input")
        output_node = Operation('dense', units=36, activation='sigmoid')

        train, test = load_deepsea_data(wd, False)
        X_train, Y_train = train
        X_test, Y_test = test

        bs = 128
        loss = 'binary_crossentropy'

    else:
        raise NotImplementedError

    metrics = Metrics() if not args[0] == 'deepsea' else RocCallback(training_data=(X_train, Y_train),
                                                                     validation_data=(X_test, Y_test))

    history_dir = os.path.join(wd, 'AMBER_results/outputs/AmberDeepSea_2')
    hist = read_history([os.path.join(history_dir, "train_history.csv")],
                        metric_name_dict={'zero': 0, 'auc': 1})
    hist = hist.sort_values(by='auc', ascending=False)
    hist.head(n=5)
    model_space = get_model_space(out_filters=32, num_layers=12)

    keras_builder = KerasResidualCnnBuilder(inputs_op=input_node,
                                            output_op=output_node,
                                            fc_units=100,
                                            flatten_mode='Flatten',
                                            model_compile_dict={
                                                'loss': loss,
                                                'optimizer': 'adam',
                                                'metrics': ['acc']
                                            },
                                            model_space=model_space,
                                            dropout_rate=0.1,
                                            wsf=2
                                            )

    best_arc = hist.iloc[0][[
        x for x in hist.columns if x.startswith('L')]].tolist()
    searched_mod = keras_builder(best_arc)
    searched_mod.summary()


    # .... Define your model here ....
    # You need to have compiled your model before calling this.
    print(get_flops())

    history = searched_mod.fit(
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
            item = np.expand_dims(item, 0)
            logits = searched_mod.predict(item)
            all_pred_prob.append(logits)

        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
        final = f1_score(all_pred, Y_test, pid_test)
        print(final)
        np.savetxt('score.txt', np.array(final))

    elif args[0] == 'deepsea':
        test_predictions = []
        for item in X_test:
            logits = searched_mod.predict(item)
            logits_sigmoid = sigmoid(logits)
            test_predictions.append(logits_sigmoid)

        test_predictions = np.concatenate(test_predictions).astype(np.float32)
        test_gts = Y_test.astype(np.int32)

        stats = calculate_stats(test_predictions, test_gts)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])

        print(mAP)
        print(mAUC)
        np.savetxt('stats.txt', np.array([mAP, mAUC]))

    with open('metrics.txt', 'w') as file:
        file.write(json.dumps(history.history))


if __name__ == '__main__':
    main()
