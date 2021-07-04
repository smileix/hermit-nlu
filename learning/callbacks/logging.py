from keras.callbacks import ProgbarLogger
from keras.utils.generic_utils import Progbar


class TuningProgbar(ProgbarLogger):

    def __init__(self, count_mode='samples',
                 stateful_metrics=None):
        super(TuningProgbar, self).__init__(count_mode, stateful_metrics)
        self.target = None
        self.progbar = None
        self.seen = None
        self.epochs = None

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        if self.use_steps:
            target = self.params['steps']
        else:
            target = self.params['samples']
        self.target = target
        self.progbar = Progbar(target=self.target,
                               verbose=1,
                               stateful_metrics=self.stateful_metrics)
        self.seen = 0

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        self.progbar.update(self.seen, self.log_values)
