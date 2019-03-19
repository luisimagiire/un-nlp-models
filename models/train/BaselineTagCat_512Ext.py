import os

import dask.dataframe as dd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input, Flatten
from keras.models import Model

from models.basekeras import KerasBaseModel, KerasConfig


class BaseTagCatConfigExt(KerasConfig):
    def __init__(self, config):
        super().__init__(config=config)
        self.num_tags = None
        self.num_cats = None
        self.num_words = None


class BaseTagCatKerasExt(KerasBaseModel):

    def __init__(self, configfile):
        self.name = "BaseTagCatExt"
        super().__init__(configfile)
        self.model_folder = self.get_model_folder()

    def get_config(self, config):
        return BaseTagCatConfigExt(config)

    def build_model_structure(self, opts=None):
        if self.model is None:
            # Multi-output model
            inputed = Input((self.model_config.sequence_lenght,), name='text_input')
            x = Embedding(input_dim=self.model_config.num_words, output_dim=self.model_config.embbeding_size)(inputed)
            x = Flatten()(x)
            x = Dense(units=512)(x)
            tags_prediction = Dense(self.model_config.num_tags, activation='sigmoid', name='tags')(x)
            category_prediction = Dense(self.model_config.num_cats, activation='softmax', name='category')(x)

            model = Model(inputed, [tags_prediction, category_prediction])
            model.compile(optimizer='adam', loss={'tags': 'categorical_crossentropy',
                                                  'category': 'categorical_crossentropy'}, metrics=['acc'])
            print(model.summary())
            self.model = model
            self.save_model_architecture()
        else:
            print("MODEL ALREADY DEFINED! USING DEFINED VERSION...")

    def train(self, dataset_folder):
        train_files, test_files, label_files = self.get_dataset_files(dataset_folder)
        self.update_label_vars(label_files)
        self.save_encoders_at_model_folder(label_files)
        train_gen = self.get_generator(train_files)
        test_gen = self.get_generator(test_files)

        self.build_model_structure()
        tensorboard = self.get_tensorboard()
        checkpoints = self.get_checkpoint_weights()
        stopping = self.get_early_stopping()

        fit_params = {"generator": train_gen,
                      "validation_data": test_gen,
                      "use_multiprocessing": True,
                      "epochs": self.model_config.epochs,
                      "steps_per_epoch": self.model_config.epochs_size // self.model_config.batch_size,
                      "workers": self.model_config.num_workers,
                      "verbose": 1,
                      "callbacks": [tensorboard, checkpoints, stopping]}

        fit_params['validation_steps'] = self.model_config.validation_size // self.model_config.batch_size

        self.history = self.model.fit_generator(**fit_params)

    def get_generator(self, path):
        df = dd.read_json(path)
        while True:
            sample_df = df.sample(frac=0.01).compute()
            x500 = np.array([np.array(x) for x in sample_df.one_hot.values])
            y500_t = np.array([np.array(y) for y in sample_df.one_hot_tags.values])
            y500_c = np.array([np.array(y) for y in sample_df.one_hot_cat.values])
            yield x500, {'tags': y500_t, 'category': y500_c}

    def get_checkpoint_weights(self):
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder,
                         "weights-improvement-{epoch:02d}-{category_acc:.2f}-{tags_acc:.2f}.hdf5"),
                         verbose=1, save_best_only=True)
        return checkpoint

    def update_label_vars(self, files):
        files.sort()
        labels = self.load_encoders(files)
        for label in labels:
            if "category" in label["name"]:
                self.model_config.num_cats = len(label["encoder"].classes_)
            elif "tags" in label["name"]:
                self.model_config.num_tags = len(label["encoder"].classes_)
            elif "tokenizer" in label["name"]:
                self.model_config.num_words = len(label["encoder"].word_index)

    def load_base(self, weights, dataset_folder):
        _, _, label_files = self.get_dataset_files(dataset_folder)
        self.update_label_vars(label_files)
        self.build_model_structure()
        self.model.load_weights(weights)

    def load(self, path):
        # Loading encoders
        labels_path = os.path.join(path, 'encoders')
        encoder_files = [os.path.join(labels_path, x) for x in os.listdir(labels_path)]
        self.update_label_vars(encoder_files)

        # Loading model structure
        self.build_model_structure()
