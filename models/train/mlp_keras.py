from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.models import Sequential

from models.basekeras import KerasBaseModel, KerasConfig


class MLPConfig(KerasConfig):
    pass


class MLP(KerasBaseModel):

    def __init__(self, config):
        self.name = "MLP"
        super().__init__(configfile=config)

    def get_config(self, config):
        return MLPConfig(config=config)

    def build_model_structure(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.model_config.input_dim))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, input_dim=self.model_config.input_dim))
        model.add(Activation('relu'))
        model.add(Dense(17))
        model.add(Activation('softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model
