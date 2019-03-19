import logging
import os
import sys
from datetime import datetime
from pprint import pprint

import numpy as np
import psutil
import yaml


def get_results(inputs, encoders, model, top_k=0.5):
    def get_top_scores(veco, perc):
        return [(i,x) for i,x in enumerate(veco) if x>perc]

    def return_vecs(predictions, encoders, top_k=0.5):
        tmp_tags = []
        tmp_cats = []
        for i,prediction in enumerate(predictions):
            tmp_encoder = encoders[i]
            if i == 0:
                for pred in prediction:
                    tmp_vec = get_top_scores(pred, top_k)
                    ad_tags = []
                    for vec_index,vec_score in tmp_vec:
                        zero = np.zeros((len(prediction[0]),1))
                        zero[vec_index] = 1
                        ad_tags.append({'pred':tmp_encoder.inverse_transform(zero.reshape(1,len(prediction[0])))[0], 'tag_score':vec_score})
                    tmp_tags.append(ad_tags)
            else:
                for pred in prediction:
                    tmp_cats.append({'pred': tmp_encoder.inverse_transform(np.array(pred).reshape(1,len(prediction[0])))[0]})
        return tmp_tags, tmp_cats

    tmp_pred = model.predict_on_batch(inputs)
    return return_vecs(tmp_pred, encoders, top_k)

def print_res(dfzao, results):
    ls_dicts = []
    for i,veco in enumerate(dfzao):
        base_dict = {'input': veco[0], 'true_tag': veco[1], 'true_cat': veco[3]}
        for j,res in enumerate(results):
            if j ==0:
                base_dict["pred_tag"] = res[i]
            else:
                base_dict['pred_cat'] = res[i]['pred']
        pprint(base_dict)
        ls_dicts.append(base_dict)



def configLog(log_path, mode=None):
    if mode is None:
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=log_path,
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=log_path,
                            level=mode)


def pre_loading(argparser, config_path):
    parser = argparser()
    parsed_args = parser.parse_args(sys.argv[1:])
    sys_variables = vars(parsed_args)
    config = read_config_file(config_path)
    return sys_variables, config


def read_config_file(config_path):
    with open(config_path, "r") as yamlconfig:
        config = yaml.load(yamlconfig)
    return config


def get_notice(perc, base=10000):
    if perc % base == 0:
        message("{} data points processed!".format(perc))
        message("Time: {}".format(datetime.ctime(datetime.now())))


def message(msg, *args, **kwargs):
    print('[*] ', msg, *args, **kwargs)
    logging.info(msg)


def get_system_status():
    message("Memory usage: {} %".format(psutil.virtual_memory().percent))
    message("CPU usage: {} %".format(psutil.cpu_percent()))


def start_logging(mode):
    tmp_counter = 0

    if not os.path.isdir('logs'):
        os.mkdir('logs')

    if not os.path.isdir('logs/{}'.format(mode)):
        os.mkdir('logs/{}'.format(mode))

    log_name = 'logs/{0}/process_{1}_{2}.log'.format(mode, str(datetime.date(datetime.utcnow())), tmp_counter)
    while os.path.isfile(log_name):
        tmp_counter += 1
        log_name = 'logs/{0}/process_{1}_{2}.log'.format(mode, str(datetime.date(datetime.utcnow())), tmp_counter)

    configLog(log_path=log_name, mode=logging.DEBUG)


def check_folder(folder_path):
    if not os.path.isdir(folder_path):
        message("{} not found! Creating folder...".format(folder_path))

        # Creating intermediate folders
        tmp_path, file_path = os.path.split(folder_path)
        while len(file_path.strip()) == 0:
            tmp_path, file_path = os.path.split(tmp_path)

        while not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
            tmp_path = os.path.split(tmp_path)[0]

        os.mkdir(folder_path)
    return folder_path

