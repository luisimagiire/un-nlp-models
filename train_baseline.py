import utils.util as sc
import argparse
from models.train.BaselineTagCat import BaseTagCatKeras
from models.train.BaselineTagCat_512Ext import BaseTagCatKerasExt
from models.train.BaselineTagCat_w2vec import BaseTagCatKerasW2V
from models.train.BaselineTagCat_fttext import BaseTagCatKerasFT
from models.train.TagSpecific_w2vec import TagKerasW2V
from models.train.PriceBaseline_w2vec import PriceKerasW2V


def argument_parser():
    parser = argparse.ArgumentParser(description='Unegocio v01 NLP Model Training')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        help='Model to train',
                        required=True)
    return parser


def main(model, config):

    if model == "BaseTagCatKeras":
        model = BaseTagCatKeras(config)
    elif model == "BaseTagCatKerasExt":
        model = BaseTagCatKerasExt(config)
    elif model == "BaseTagCatKerasW2V":
        model = BaseTagCatKerasW2V(config)
        pass
    elif model == "BaseTagCatKerasFT":
        model = BaseTagCatKerasFT(config)
    elif model == "TagKerasW2V":
        model = TagKerasW2V(config)
    elif model == "PriceKerasW2V":
        model = PriceKerasW2V(config)
    else:
        raise Exception("Model {} does not exist!".format(model))

    model.train(config["GENERAL"]["dataset_path"])
    model.save()


if __name__ == '__main__':
    sys_vars, config_set = sc.pre_loading(argument_parser, 'config.yaml')
    execution_mode = sys_vars['model']
    main(execution_mode, config_set)

