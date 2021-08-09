import tensorflow as tf
from tensorflow import keras
import amber
import os
import shutil
from amber.modeler import KerasResidualCnnBuilder
from amber.utils.io import read_history

from load_ecg import read_data_physionet_4, custom_f1, f1_score
from load_satellite import load_satellite_data 

def main():
    args = sys.argv[1:]
    wd = '.'
    if args[0] == 'ECG':
        input_node = Operation('input', shape=(3000, 1), name="input")
        output_node = Operation('dense', units=4, activation='sigmoid')
        
        X_train, Y_train, X_test, Y_test, pid_test = read_data_physionet_4('.')
        bs = 128

    elif args[0] == 'satellite':
        input_node = Operation('input', shape=(46, 1), name="input")
        output_node = Operation('dense', units=24, activation='sigmoid')

        X_train, Y_train, X_test, Y_test = load_satellite_data('.', False)
        bs = 1024

    else:
        raise NotImplementedError

    
    hist = read_history([os.path.join(wd, "train_history.csv")], 
                        metric_name_dict={'zero':0, 'auc': 1})
    hist = hist.sort_values(by='auc', ascending=False)
    hist.head(n=5)

    keras_builder = KerasResidualCnnBuilder(inputs_op=input_node,
        output_op=output_node,
        fc_units=100,
        flatten_mode='Flatten',
        model_compile_dict={
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam',
            'metrics': ['acc', custom_f1]
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
        callbacks=[history]
    )

    if args[0] == 'ECG':
        all_pred_prob = []
        for item in X_test:
            logits = searched_mod.predict(item)
            all_pred_prob.append(logits.eval())

        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)
        final= f1_score(all_pred, Y_test, pid_test)
        print(final)
        np.savetxt('score.txt', final)
    
    np.savetxt('metrics.txt',history.history)

