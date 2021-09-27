try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K
import numpy as np
import os
import matplotlib.pyplot as plt
from .utils import moving_window_avg

'''
https://github.com/psklight/keras_one_cycle_clr/tree/master/keras_one_cycle_clr
'''
class LrRangeTest(keras.callbacks.Callback):
    """
    A callback class for finding a learning rate.
    :param lr_range: a tuple of lower and upper bounds of learning rate.
    :param wd_list: a list of weight decay to perform grid search.
    :param steps: a number of steps for learning rates in a range test.
    :param batches_per_step: a number of batches to average loss for each learning rate step.
    :param threshold_multiplier: a multiplier to lowest encountered training loss to determine early termination of range test.
    :param validation_data: either (x_test, y_test) or a generator. Useful for wd grid search.
    :param validation_batch_size: a batch size when evaluating validation loss.
    :param batches_per_val: a number of batches to use in averaging validation loss.
    :param verbose: True or False whether to print out progress detail.
    """

    def __init__(self,
                 lr_range=(1e-5, 10),
                 wd_list=[],
                 steps=100,
                 batches_per_step=5,
                 threshold_multiplier=5,
                 validation_data=None,
                 validation_batch_size=16,
                 batches_per_val=10,
                 verbose=False):

        super(LrRangeTest, self).__init__()

        self.lr_range = lr_range

        self.wd_list = wd_list

        self.steps = steps
        self.batches_per_step = batches_per_step
        self.early_stop = False
        self.threshold_multiplier = threshold_multiplier
        self.validation_data = validation_data
        if validation_data is not None:
            self.use_validation = True
        else:
            self.use_validation = False
        self.validation_batch_size = validation_batch_size
        self.batches_per_val = batches_per_val
        self.verbose = verbose

        # generate a range of learning rates
        self.lr_values = np.power(10.0,
                                  np.linspace(np.log10(lr_range[0]), np.log10(lr_range[1]), self.steps))

        # logs initialization
        self.lr = self.lr_values
        n_wd = len(self.wd_list) if len(self.wd_list) > 0 else 1
        self.loss = np.zeros(shape=(self.lr_values.size, n_wd)) * np.nan
        if self.use_validation:
            self.val_loss = np.zeros_like(self.loss) * np.nan

        # non-reset counters
        self.current_wd = 0
        self.model_org = []

        # reset counters
        self.current_batches_per_step = 0
        self.current_loss_val = 0

    def _fetch_val_batch(self, batch):
        if isinstance(self.validation_data, (tuple,)):
            batch_size = self.validation_batch_size
            x = self.validation_data[0][batch * batch_size:(batch + 1) * batch_size]
            y = self.validation_data[1][batch * batch_size:(batch + 1) * batch_size]
            return x, y
        if isinstance(self.validation_data, (keras.utils.Sequence,)):
            return self.validation_data.__getitem__(batch)

    def _reset(self):
        """
        Reset counters, prepare for a new weight decay value.
        """
        self.model.optimizer.set_weights(self.model_org.optimizer.get_weights())
        self.model.set_weights(self.model_org.get_weights())
        self.current_step = 0
        self.current_batches_per_step = 0
        self.current_loss_val = 0
        self.best_loss = np.inf
        self.early_stop = False

    def on_train_begin(self, logs={}):
        # save current model for reset
        self.model.save("lr_range_test_original_stage.h5")
        self.model_org = keras.models.load_model("lr_range_test_original_stage.h5")
        # handle empty input wd_list
        if len(self.wd_list) == 0:
            self.wd_list = [K.get_value(self.model.optimizer.decay)]
        self.current_wd = 0
        self._reset()

    def on_train_batch_begin(self, batch, logs):
        K.set_value(self.model.optimizer.lr, self.lr_values[self.current_step])
        K.set_value(self.model.optimizer.decay, self.wd_list[self.current_wd])

    def on_train_batch_end(self, batch, logs):

        self.current_loss_val += logs['loss']
        self.current_batches_per_step += 1

        if self.current_batches_per_step == self.batches_per_step:

            self.loss[self.current_step, self.current_wd] = self.current_loss_val / self.batches_per_step

            if self.use_validation:
                # calculate for validation set
                self.current_loss_val = 0.0
                if isinstance(self.validation_data, tuple):
                    batch_size = self.validation_batch_size
                    N = int(np.ceil(self.validation_data[0].shape[0] / batch_size))
                if isinstance(self.validation_data, keras.utils.Sequence):
                    N = len(self.validation_data)
                n_batch = min(self.batches_per_val, N)
                for i in range(n_batch):
                    data_batch = self._fetch_val_batch(i)
                    batch_size = data_batch[0].shape[0]
                    result = self.model.evaluate(x=data_batch[0], y=data_batch[1],
                                                 batch_size=batch_size,
                                                 verbose=False)
                    self.current_loss_val += result[0]

                self.val_loss[self.current_step, self.current_wd] = self.current_loss_val / n_batch

            # verbose
            if self.verbose:
                if not self.use_validation:
                    print("wd={:.2e}".format(self.wd_list[self.current_wd]), ",",
                          "lr={:.2e}".format(self.lr_values[self.current_step]), ",",
                          "loss={:.2e}".format(self.loss[self.current_step - 1, self.current_wd]))
                if self.use_validation:
                    print("wd={:.2e}".format(self.wd_list[self.current_wd]), ",",
                          "lr={:.2e}".format(self.lr_values[self.current_step]), ",",
                          "loss={:.2e}".format(self.loss[self.current_step - 1, self.current_wd]), ",",
                          "val_loss={:.2e}".format(self.val_loss[self.current_step - 1, self.current_wd]))

            self.current_batches_per_step = 0
            self.current_loss_val = 0.0
            self.current_step += 1

            # update best loss
            if not self.use_validation:
                latest_loss = self.loss[self.current_step - 1, self.current_wd]
            else:
                latest_loss = self.val_loss[self.current_step - 1, self.current_wd]

            self.best_loss = self.best_loss if self.best_loss < latest_loss else latest_loss

            # determine earlystop
            if latest_loss > self.best_loss * self.threshold_multiplier:
                self.early_stop = True

        # consider next wd value
        if self.current_step == self.lr_values.size or self.early_stop:
            self.current_wd += 1
            self._reset()

        # stop training when done with all weight decays, set everything back to before lr range test.
        if self.current_wd == len(list(self.wd_list)):
            self.model.set_weights(self.model_org.get_weights())
            K.set_value(self.model.optimizer.lr,
                        K.get_value(self.model_org.optimizer.lr))
            self.model.optimizer.set_weights(self.model_org.optimizer.get_weights())
            self.model.stop_training = True
            try:
                os.remove("lr_range_test_original_stage.h5")
            except:
                pass

    def find_n_epoch(self, dataset, batch_size=None):
        """
        A method to find a number of epochs to train in the sweep.
        :param dataset: If the training data is an ndarray (used with model.fit), ``dataset`` is the x_train. If the training data is a generator (used with model.fit_generator), ``dataset`` is the generator instance.
        :param batch_size: Needed only if ``dataset`` is x_train.
        :return epochs: a number of epochs needed to do a learning rate sweep.
        """
        n_wd = len(self.wd_list) if len(self.wd_list) > 0 else 1
        if isinstance(dataset, keras.utils.Sequence):
            return int(np.ceil(self.steps * self.batches_per_step / len(dataset)) * n_wd)
        if isinstance(dataset, np.ndarray):
            if batch_size is None:
                raise ValueError("``batch_size`` must be provided.")
            else:
                return int(np.ceil(self.steps * self.batches_per_step /
                                   (dataset.shape[0] / batch_size)) * n_wd)

    def plot(self, set='train', x_scale="log", y_scale="linear", ma=True, window=5, **kwargs):
        """
        Plot the lr range test result.
        :param set: either "train" or "valid". If 'valid', ``validation_data`` must not be ``None``.
        :param x_scale: scale for the x axis, either "log" or "linear".
        :param y_scale: scale for the y axis, either "log" or "linear".
        :param ma: True or False to use moving windonw average.
        :param window: an integer for window of averaging.
        :param kwargs: valid ``kwargs`` to [``pyplot.plot``](https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.plot.html) function.
        """

        assert set in ["train", "valid"], "``set`` must be either ""train"" or ""test""."
        if set is "valid" and not self.use_validation:
            raise ValueError("There is not validation data used to plot. Change set to ""train"".")
        assert x_scale in ["log", "linear"], "x_scale must be either ""log"", or ""linear""."
        assert y_scale in ["log", "linear"], "y_scale must be either ""log"", or ""linear""."

        plt.figure()

        n_wd = len(self.wd_list) if len(self.wd_list) > 0 else 1

        if set is "valid":
            loss = self.val_loss
            y_str = "val loss"
        if set is "train":
            loss = self.loss
            y_str = "train loss"

        if ma:
            loss = np.copy(loss) # prevent overriding.
            for i in range(n_wd):
                loss[:, i] = moving_window_avg(loss[:, i], window=window)

        # build legend
        legends = []
        for w in self.wd_list:
            legends.append("wd={:.1e}".format(w))

        lr = self.lr
        plt.plot(lr, loss, **kwargs)
        plt.xlabel("lr")
        plt.ylabel(y_str)
        plt.xscale(x_scale)
        plt.yscale(y_scale)
        plt.legend(tuple(legends))
        plt.show()
