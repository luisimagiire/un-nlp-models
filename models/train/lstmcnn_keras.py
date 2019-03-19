from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from numpy import newaxis

from models.basekeras import KerasBaseModel, KerasConfig

class LSTMCNNConfig(KerasConfig):
    def __init__(self, config):
        super().__init__(config=config)
        self.custom_shaper = lambda vec: vec[:, newaxis, :]  # Will reshape input vector to fit lstm layer


class LSTMCNNKeras(KerasBaseModel):

    def __init__(self, configfile):
        self.name = "LSTM"
        super().__init__(configfile)

    def get_config(self, config):
        return LSTMCNNConfig(config)

    def build_model_structure(self):
        model = Sequential()
        model.add(Conv1D(filters=32, input_shape=( 1, self.model_config.input_dim), kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128))
        # model.add(Dropout(0.5))
        model.add(Dense(17, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model
