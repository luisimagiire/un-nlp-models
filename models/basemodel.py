import logging
import os
import pickle
from collections import namedtuple
from datetime import datetime
from time import time

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from utils import util as sc
from shutil import copytree


class BaseModel:
    name = "BaseModel"
    data_handler = namedtuple("DataHandler", ["data_partitions", "labels"])

    def __init__(self, configfile):
        log_path = sc.check_folder(configfile["TRAINING"]['log_path'])
        log_path = os.path.join(log_path, '{0}_{1}_train.log'.format(self.name, time()))
        sc.configLog(log_path=log_path, mode=logging.DEBUG)
        self.configfile = configfile
        self.model_config = self.get_config(configfile)
        self.encoder = None
        self.embedder = None
        self.files_handler = None
        self.model = None
        self.test_set = None
        self.model_folder = None

    def load(self, path):
        pass

    def validate(self):
        pass

    def get_config(self, configfile):
        # Override to set model's custom config
        return BaseConfig(configfile)

    def train(self, data):
        pass

    def save(self):
        pass

    def build_model_structure(self, opts=None):
        pass

    def __str__(self):
        return self.name

    def get_model_folder(self):
        """
        Gives a new folder for the trained model.
        :return: (str) folder to save the model
        """
        # Creating model folder
        today_dt = str(datetime.date(datetime.utcnow()))
        model_folder = sc.check_folder(os.path.join(
            self.model_config.output_path,
            self.name,
            today_dt))

        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        else:
            tmp_count = 0
            # Get next nonexistent folder
            while os.path.isdir(os.path.join(self.model_config.output_path,
                                             self.name, '{0}_{1}'.format(tmp_count, today_dt))):
                tmp_count += 1

            os.mkdir(os.path.join(self.model_config.output_path, self.name,
                                  '{0}_{1}'.format(tmp_count, today_dt)))
            model_folder = os.path.join(self.model_config.output_path, self.name,
                                        '{0}_{1}'.format(tmp_count, today_dt))
        return model_folder

    def load_encoders(self, labels):
        """
        Load encoders from pickle or build them from numpy array.
        :param labels: list with encoders path
        :return: list of dicts with name and the encoder object
        """
        encoders = []
        for label_path in labels:
            if label_path.endswith('.pckl'):
                with open(label_path, 'rb') as dt:
                    encoders.append({'name': os.path.split(label_path)[-1],
                                     'encoder': pickle.load(dt)})
            # else:
            #     tmp_encoder = LabelBinarizer()
            #     tmp_encoder.fit(np.load(label_path))
            #     encoders.append({'name': os.path.split(label_path)[-1],
            #                      'encoder': tmp_encoder})
        return encoders

    @staticmethod
    def get_dataset_files(dataset_folder):
        """
        Given datasets/preprocess/nlp folder, gets training/validation csv files and label list.
        :param dataset_folder: (str) Nlp folder path
        :return: (dict) train/validation split paths
        """
        def validate_paths(path):
            if not os.path.isdir(path):
                raise Exception("Folder {} is not a valid path!".format(path))

            partitions = [os.path.join(path, file)
                          for file in os.listdir(path)
                          if any([file.endswith(x) for x in ['.csv', '.npy', '.json', '.pckl']])]

            if len(partitions) == 0:
                raise FileNotFoundError("No valid files were found @ {}".format(data_path))
            return partitions

        # Get dataset files
        data_path = validate_paths(os.path.join(dataset_folder, "train"))
        label_path = validate_paths(os.path.join(dataset_folder, "labels"))
        test_path = validate_paths(os.path.join(dataset_folder, "test"))

        return data_path, test_path, label_path

    def save_encoders_at_model_folder(self, encoders_path):
        if len(encoders_path) < 1:
            raise ValueError("No encoders were found @ {}".format(encoders_path))
        encoder_folder = os.path.split(encoders_path[0])[-2]
        copytree(encoder_folder, os.path.join(self.model_folder, 'encoders'))



class BaseConfig:
    batch_size = None
    num_workers = None
    epochs_size = None
    output_path = None
    TENSORBOARD = None
    only_train_gen = True
    custom_shaper = None

    def __init__(self, configfile):
        self.unpack_config(configfile)
        self.params = self.parameters()

    def parameters(self):
        pass

    def unpack_config(self, configfile):
        for name, value in configfile["TRAINING"].items():
            if name == 'output_path':
                value = sc.check_folder(value)
            setattr(self, name, value)

    def save_params(self, folder):
        sc.check_folder(folder)
        with open(os.path.join(folder, 'params.txt'), 'w') as f:
            f.write("Trained parameters: \n")
            f.write(str(self.params))
