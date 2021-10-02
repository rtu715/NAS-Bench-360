import os
from os import system, listdir
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Lambda,
    Permute,
    Multiply,
    BatchNormalization,
)

from activations import Mish
from optimizers import Ranger
import losses as l
import callbacks as cb
from layers import Attention, LayerNormalization
from data import dataset
from generator import generator

import keras.backend as K
def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
				run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops #Prints the "flops" of the model.

tf.random.set_random_seed(2)
# download the data
if "ninaPro" not in listdir():
    system('wget -c https://www.dropbox.com/s/kxrqhqhcz367v77/nina.tar.gz?dl=1 -O - | tar -xz')

# read in the data
'''
data = dataset("./ninaPro")

reps = np.unique(data.repetition)
val_reps = reps[3::2]
train_reps = reps[np.where(np.isin(reps, val_reps, invert=True))]
test_reps = val_reps[-1].copy()
val_reps = val_reps[:-1]
train = generator(data, list(train_reps), imu = imu)
validation = generator(data, list(val_reps), augment=False, imu = imu)
test = generator(data, [test_reps][0], augment=False, imu = imu)
'''
path = './my_ninapro'
train_data = np.load(os.path.join(path, 'ninapro_train.npy'),
                             encoding="bytes", allow_pickle=True)
#train_data = train_data[:, None, :, :]
train_labels = np.load(os.path.join(path, 'label_train.npy'), encoding="bytes", allow_pickle=True)

valid_data = np.load(os.path.join(path, 'ninapro_val.npy'),
                             encoding="bytes", allow_pickle=True)
#valid_data = valid_data[:, None, :, :]
valid_labels = np.load(os.path.join(path, 'label_val.npy'), encoding="bytes", allow_pickle=True)

test_data = np.load(os.path.join(path, 'ninapro_test.npy'),
                             encoding="bytes", allow_pickle=True)
#test_data = test_data[:, None, :, :]
test_labels = np.load(os.path.join(path, 'label_test.npy'), encoding="bytes", allow_pickle=True).astype(np.int8)

train_data = np.concatenate((train_data, valid_data), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0).astype(np.int8)
train_labels = np.eye(18)[train_labels]
test_labels = np.eye(18)[test_labels]
# model parameters
timesteps = train_data[0].shape[0]
n_class = 18
n_features = train_data[0].shape[-1] # 16 channels
model_pars = {
    "timesteps": timesteps,
    "n_class": n_class,
    "n_features": n_features,
    "classifier_architecture": [500, 500, 2000],
    "dropout": [0.36, 0.36, 0.36],
}

train = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
test = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

batch_size = 128
train = train.shuffle(100).batch(batch_size)
test = test.batch(batch_size)

# attention mechanism
def attention_simple(inputs, timesteps):
    input_dim = int(inputs.shape[-1])
    a = Permute((2, 1), name='transpose')(inputs)
    a = Dense(timesteps, activation='softmax',  name='attention_probs')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply(name='focused_attention')([inputs, a_probs])
    output_flat = Lambda(lambda x: K.sum(x, axis=1), name='temporal_average')(output_attention_mul)
    return output_flat, a_probs


def dense_model(timesteps, n_class, n_features, classifier_architecture, dropout):
    inputs = Input((timesteps, n_features))
    x = BatchNormalization(axis = 1, momentum = 0.99)(inputs)
    x = Dense(128, activation=Mish())(x)
    x = LayerNormalization()(x)
    x, a = attention_simple(x, timesteps)
    for d, dr in zip(classifier_architecture, dropout):
        x = Dropout(dr)(x)
        x = Dense(d, activation=Mish())(x)
        x = LayerNormalization()(x)
    outputs = Dense(n_class, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model



model = dense_model(**model_pars)

cosine = cb.CosineAnnealingScheduler(
    T_max=200, eta_max=1e-3, eta_min=1e-5, verbose=1, epoch_start=5
)
loss = l.focal_loss(gamma=3., alpha=6.)
#loss = 'sparse_categorical_crossentropy'
model.compile(Ranger(learning_rate=1e-3), loss=loss, metrics=["accuracy"])

print(model.summary())
print(get_flops())

model.fit(
    train,
    epochs=205,
    #validation_data=validation,
    callbacks=[
        ModelCheckpoint(
            "main.h5",
            #monitor="val_loss",
            keep_best_only=True,
            save_weights_only=False,
        ),
        cosine,
    ],
    shuffle = False,
)


#model.evaluate(validation)
model.evaluate(test)

