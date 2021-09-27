import math
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0, epoch_start=80, restart_epochs=None, gamma=1, expansion=1, flat_end = False):
        super(CosineAnnealingScheduler, self).__init__()
        self.epoch_start=epoch_start
        self.expansion=expansion
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.restart_epochs = restart_epochs
        self.gamma = gamma
        self.flat_end = flat_end

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        if epoch > self.epoch_start - 1:
            if self.restart_epochs is None:
                learning_rate = self.eta_min + (self.eta_max*self.gamma - self.eta_min) * (1 + math.cos(math.pi * (epoch - self.epoch_start) / self.T_max)) / 2
                K.set_value(self.model.optimizer.learning_rate, learning_rate)
            else:
                learning_rate = self.eta_min + (self.eta_max*self.gamma - self.eta_min) * (1 + math.cos(math.pi * ((epoch  % (self.restart_epochs+self.epoch_start)) - self.epoch_start) / self.T_max)) / 2
                K.set_value(self.model.optimizer.learning_rate, learning_rate)
            if learning_rate<=self.eta_min:
                self.eta_max *= self.gamma
                self.T_max *=self.expansion
        if self.flat_end and epoch >= ((self.epoch_start -1 ) + T_max):
            learning_rate = self.eta_min

        else:
            learning_rate=self.model.optimizer.learning_rate
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, learning_rate))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.learning_rate)

