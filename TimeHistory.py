from keras.callbacks import Callback
import time

# from https://stackoverflow.com/a/43186440

class TimeHistory(Callback):
    def __init__(self):
        self.times = []

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def get_times(self):
        return self.times