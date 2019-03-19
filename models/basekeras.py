
import os
import pickle
from time import time

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard

import utils.util as sc
from models.basemodel import BaseModel, BaseConfig


class KerasConfig(BaseConfig):

    def __init__(self, config):
        super().__init__(configfile=config)


class KerasBaseModel(BaseModel):

    def __init__(self, configfile):
        super().__init__(configfile)
        self.history = None

    def get_config(self, config):
        return KerasConfig(config)

    def save(self):
        sc.message("Saving model...")
        model_folder = self.model_folder
        self.model.save(os.path.join(model_folder, "{0}_{1}.h5".format(self.name, time())))

        # Saves trainning history
        if self.history is not None:
            with open(os.path.join(model_folder, 'training_data.pkl'), 'wb') as output:
                pickle.dump(self.history.history, output, pickle.HIGHEST_PROTOCOL)
            sc.message("Training History saved!")

    def save_model_architecture(self):
        model_json = self.model.to_json()
        with open(os.path.join(self.model_folder, "model_arc.json"), "w") as json_file:
            json_file.write(model_json)
        sc.message('Model architecture saved!')

    def get_checkpoint_weights(self):
        checkpoint = ModelCheckpoint(os.path.join(self.model_folder,
                                                  "weights-improvement-{epoch:02d}.hdf5"),
                                     verbose=self.configfile["TRAINING"]["CheckPoint"]['verbose'],
                                     save_best_only=self.configfile["TRAINING"]["CheckPoint"]['save_best_only'])
        return checkpoint

    def get_early_stopping(self):
        stopping = EarlyStopping(monitor=self.configfile["TRAINING"]["EarlyStopping"]['monitor'],
                                 min_delta=self.configfile["TRAINING"]["EarlyStopping"]['min_delta'],
                                 patience=self.configfile["TRAINING"]["EarlyStopping"]['patience'],
                                 verbose=self.configfile["TRAINING"]["EarlyStopping"]['verbose'],
                                 mode=self.configfile["TRAINING"]["EarlyStopping"]['mode'])
        return stopping

    def get_tensorboard(self):
        """
        Build Tensorboard keras callback.
        :return: TB callback
        """
        tensorboard_path = sc.check_folder(os.path.join(self.model_folder, "tensorboard"))
        return TensorBoard(log_dir=tensorboard_path, histogram_freq=0)
