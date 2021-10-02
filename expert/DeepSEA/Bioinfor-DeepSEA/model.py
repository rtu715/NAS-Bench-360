# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

class DeepSEA(keras.Model):
    def __init__(self):
        super(DeepSEA, self).__init__()
        self.conv_1 = keras.layers.Conv1D(
            filters=320,
            kernel_size=8,
            strides=1,
            use_bias=False,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.pool_1 = keras.layers.MaxPool1D(
            pool_size=4,
            strides=4,
            padding='SAME')

        self.dropout_1 = keras.layers.Dropout(0.2)

        self.conv_2 = keras.layers.Conv1D(
            filters=480,
            kernel_size=8,
            strides=1,
            use_bias=False,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.pool_2 = keras.layers.MaxPool1D(
            pool_size=4,
            strides=4,
            padding='SAME')

        self.dropout_2 = keras.layers.Dropout(0.2)

        self.conv_3 = keras.layers.Conv1D(
            filters=960,
            kernel_size=8,
            strides=1,
            use_bias=False,
            padding='SAME',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.dropout_3 = keras.layers.Dropout(0.5)

        self.flatten = keras.layers.Flatten()

        self.dense_1 = keras.layers.Dense(
            units=925,
            use_bias=False,
            activation='relu',
            activity_regularizer=tf.keras.regularizers.l1(1e-08),
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))

        self.dense_2 = keras.layers.Dense(
            units=36,
            use_bias=False,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(5e-07),
            kernel_constraint=tf.keras.constraints.MaxNorm(0.9))


    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward propagation of DeepSEA model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 1000, 4]
        # Output Tensor Shape: [batch_size, 1000, 320]
        temp = self.conv_1(inputs)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 1000, 320]
        # Output Tensor Shape: [batch_size, 250, 320]
        temp = self.pool_1(temp)

        # Dropout Layer 1
        temp = self.dropout_1(temp, training = training)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 250, 320]
        # Output Tensor Shape: [batch_size, 250, 480]
        temp = self.conv_2(temp)

        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 250, 480]
        # Output Tensor Shape: [batch_size, 63, 480]
        temp = self.pool_2(temp)

        # Dropout Layer 2
        temp = self.dropout_2(temp, training = training)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 63, 480]
        # Output Tensor Shape: [batch_size, 63, 960]
        temp = self.conv_3(temp)

        # Dropout Layer 3
        temp = self.dropout_3(temp, training = training)

        # Flatten Layer 1
        # Input Tensor Shape: [batch_size, 63, 960]
        # Output Tensor Shape: [batch_size, 60480]
        temp = self.flatten(temp)

        # Fully Connection Layer 1
        # Input Tensor Shape: [batch_size, 60480]
        # Output Tensor Shape: [batch_size, 925]
        temp = self.dense_1(temp)

        # Fully Connection Layer 2
        # Input Tensor Shape: [batch_size, 925]
        # Output Tensor Shape: [batch_size, 919]
        output = self.dense_2(temp)

        return output
