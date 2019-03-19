import os

import dask.dataframe as dd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, concatenate
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from models.basekeras import KerasBaseModel, KerasConfig
import gensim


class PriceW2VConfig(KerasConfig):
    def __init__(self, config):
        super().__init__(config=config)
        self.num_tags = None
        self.num_words = None


class PriceKerasW2V(KerasBaseModel):

    def __init__(self, configfile):
        self.name = "PriceW2V"
        super().__init__(configfile)
        self.model_folder = self.get_model_folder()

    def get_config(self, config):
        return PriceW2VConfig(config)

    def build_model_structure(self, opts=None):
        # Adjust number of words
        self.model_config.num_words = self.model_config.num_words + 1

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((self.model_config.num_words, self.model_config.embbeding_size))
        for word, i in opts['tokenizer']['encoder'].word_index.items():
            if word in opts['word2vec'].wv.vocab:
                embedding_matrix[i] = opts['word2vec'][word]

        # Multi-output model
        inputed = Input((self.model_config.sequence_lenght,), name='text_input')
        auxiliary_input = Input((self.model_config.num_tags,), name='tags')

        x = Embedding(input_dim=self.model_config.num_words,
                      output_dim=self.model_config.embbeding_size,
                      weights=[embedding_matrix],
                      trainable=self.model_config.embbeding_trainable)(inputed)
        flatou = Flatten()(x)
        x = concatenate([flatou, auxiliary_input])

        x = Dense(units=128)(x)
        price_prediction = Dense(1, kernel_initializer="normal", name='price')(x)

        model = Model(inputs=[inputed, auxiliary_input], outputs=[price_prediction])
        model.compile(optimizer='adam', loss={'price': 'mean_squared_error'}, metrics=['acc'])
        print(model.summary())
        self.model = model

    def train(self, dataset_folder):
        train_files, test_files, label_files = self.get_dataset_files(dataset_folder)
        self.update_label_vars(label_files)
        train_gen = self.get_generator(train_files)
        test_gen = self.get_generator(test_files)

        self.build_model_structure(opts={'tokenizer': self.get_word_tokenizer(label_files),
                                         'word2vec': self.build_word2vec_model(dataset_folder)})
        tensorboard = self.get_tensorboard()
        checkpoints = self.get_checkpoint_weights()

        fit_params = {"generator": train_gen,
                      "validation_data": test_gen,
                      "use_multiprocessing": True,
                      "epochs": self.model_config.epochs,
                      "steps_per_epoch": self.model_config.epochs_size // self.model_config.batch_size,
                      "workers": self.model_config.num_workers,
                      "verbose": 1,
                      "callbacks": [tensorboard, checkpoints]}

        fit_params['validation_steps'] = self.model_config.validation_size // self.model_config.batch_size

        self.history = self.model.fit_generator(**fit_params)

    def build_word2vec_model(self, dataset_folder):
        labels_folder = os.path.join(dataset_folder, 'labels')
        model_path = [mod for mod in os.listdir(labels_folder) if 'word2vec' in mod]
        if len(model_path) == 0:
            raise Exception("No word2vec model was found @ labels folder...")
        print(os.path.join(labels_folder, model_path[0]))
        model = gensim.models.Word2Vec.load(os.path.join(labels_folder, model_path[0]))
        return model

    def get_generator(self, path):
        df = dd.read_json(path)
        while True:
            sample_df = df.sample(frac=0.01).compute()
            x500 = np.array([np.array(x) for x in sample_df.one_hot.values])
            x500_t = np.array([np.array(y) for y in sample_df.one_hot_tags.values])
            y500_t = np.array([np.array(y) for y in sample_df.price.values])
            yield {'text_input':x500,'tags':x500_t }, {'price': y500_t}

    def get_checkpoint_weights(self):
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_folder,
                         "weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"),
                         verbose=1, save_best_only=True)
        return checkpoint

    def update_label_vars(self, files):
        files.sort()
        labels = self.load_encoders(files)
        for label in labels:
            if "tags" in label["name"]:
                self.model_config.num_tags = len(label["encoder"].classes_)
            elif "tokenizer" in label["name"]:
                self.model_config.num_words = len(label["encoder"].word_index)

    def get_word_tokenizer(self, files):
        files.sort()
        labels = self.load_encoders(files)
        for label in labels:
            if "tokenizer" in label["name"]:
                return label

    def load(self, path):
        # Loading encoders
        labels_path = os.path.join(path, 'encoders')
        encoder_files = [os.path.join(labels_path, x) for x in os.listdir(labels_path)]
        self.update_label_vars(encoder_files)

        # Loading model structure
        self.build_model_structure()


