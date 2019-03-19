from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from numpy import newaxis

from models.basekeras import KerasBaseModel, KerasConfig

class LSTMConfig(KerasConfig):
    def __init__(self, config):
        super().__init__(config=config)
        self.custom_shaper = lambda vec: vec[:, newaxis, :]  # Will reshape input vector to fit lstm layer


class LSTMKeras(KerasBaseModel):

    def __init__(self, configfile):
        self.name = "LSTM"
        super().__init__(configfile)

    def get_config(self, config):
        return LSTMConfig(config)

    def build_model_structure(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=( 1, self.model_config.input_dim)))
        model.add(Dropout(0.5))
        model.add(Dense(17, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model
