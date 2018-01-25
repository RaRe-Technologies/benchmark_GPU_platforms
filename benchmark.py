#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a Bidirectional LSTM on the Twitter sentiment classification task.
Most of the code is directly borrowed from
https://github.com/minimaxir/keras-cntk-benchmark/blob/master/v2/test_files/imdb_bidirectional_lstm.py.

"""
from __future__ import print_function
import numpy as np
import sys
import csv
import os
import time
from keras import backend as K
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, CuDNNLSTM, Bidirectional
from keras.utils import multi_gpu_model
import random as rn
import tensorflow as tf
import logging
import argparse
from subprocess import check_output
import json
import GPUtil
import multiprocessing

from data_helpers import load_data

# Set seeds for reproducibility.
np.random.seed(123)
rn.seed(123)
tf.set_random_seed(123)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = 'results/'
EPOCH_STATS_LOGFILE = 'epoch_stats.log'
GPU_USAGE_FILE = RESULTS_DIR + 'gpu_stats.log'

class EpochStatsLogger(Callback):

    def on_train_begin(self, logs={}):
        filename = os.path.basename(sys.argv[0])[:-3]
        backend = K.backend()
        self.f = open(EPOCH_STATS_LOGFILE, 'w')
        self.log_writer = csv.writer(self.f)
        self.log_writer.writerow(['epoch', 'elapsed', 'loss',
                                  'acc', 'val_loss', 'val_acc'])

    def on_train_end(self, logs={}):
        self.f.close()

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.log_writer.writerow([epoch, time.time() - self.start_time,
                                  logs.get('loss'),
                                  logs.get('acc'),
                                  logs.get('val_loss'),
                                  logs.get('val_acc')])


def get_model():
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(CuDNNLSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    if K.backend() == 'tensorflow' and len(K.tensorflow_backend._get_available_gpus()) > 1:
        logger.info("Using Multi-GPU Model")
        model = multi_gpu_model(model, gpus=len(K.tensorflow_backend._get_available_gpus()))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--platform', help='Platform the benchmark is being run on. eg. aws, paperspace', required=True
    )
    return parser.parse_args()


def get_gpu_info():
    """Get gpu information.
    """
    gpuinfo = check_output('nvidia-smi -q', shell=True).strip()
    gpuinfo = gpuinfo.replace(':', '\n').split('\n')
    gpuinfo = [x.strip() for x in gpuinfo]
    gpuinfo_str = 'GPU INFO\n'
    gpuinfo_str += 'Model Name : {}, '.format(gpuinfo[gpuinfo.index('Product Name') + 1])
    gpuinfo_str += 'Total FB Memory : {}, '.format(gpuinfo[gpuinfo.index('FB Memory Usage') + 2])
    gpuinfo_str += 'Attached GPUs : {}, '.format(gpuinfo[gpuinfo.index('Attached GPUs') + 1])
    cuda_version = check_output('cat /usr/local/cuda/version.txt', shell=True).strip()
    gpuinfo_str += 'CUDA Version : {}'.format(cuda_version)
    return gpuinfo_str


def get_cpu_info():
    """Get system processor information.
    """
    info = check_output('lscpu', shell=True).strip().split('\n')
    cpuinfo = [l.split(":") for l in info]
    cpuinfo = [(t[0], t[1].strip()) for t in cpuinfo]
    cpuinfo = dict(cpuinfo)

    # get system memory information
    info = check_output('cat /proc/meminfo', shell=True).strip().split('\n')
    meminfo = [l.split(":") for l in info]
    meminfo = [(t[0], t[1].strip()) for t in meminfo]
    cpuinfo.update(dict(meminfo))

    info_keys = ['Model name', 'Architecture', 'CPU(s)', 'MemTotal']
    machine_info = 'CPU INFO\n'
    for k in info_keys:
        machine_info += '{}:{}, '.format(k, cpuinfo[k])
    return machine_info


def check_gpu():
    try:
        check_output('nvidia-smi', shell=True)
        return 1
    except:
        raise RuntimeError('Make sure the Docker is correctly configured for GPU usage.')


def monitor_gpu(interval, output_file):
    usages = []
    while True:
        gpu = GPUtil.getGPUs()[0]
        usage = gpu.load*100
        usages.append(usage)
        time.sleep(interval)
        with open(output_file, 'w') as f:
            for usage in usages:
                f.write("%s\n" % usage)


if __name__ == '__main__':

    options = parse_args()
    # check if GPU is correctly configured
    check_gpu()

    REPORT_FILE = '{}{}-report.json'.format(RESULTS_DIR, options.platform)

    report_dict = dict()
    # store system information
    report_dict['systeminfo'] = get_cpu_info()
    report_dict['gpuinfo'] = get_gpu_info()

    logger.info('Loading data...')
    x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv = load_data(0.01)
    logger_callback = EpochStatsLogger()

    max_features = len(vocabulary)
    maxlen = x_train.shape[1]
    batch_size = 32
    epochs = 2

    logger.info('%d train sequences' % len(x_train))
    logger.info('%d test sequences' % len(x_test))

    logger.info('x_train shape: %s' % str(x_train.shape))
    logger.info('x_test shape: %s' % str(x_test.shape))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = get_model()

    # Process to monitor GPU usage
    p = multiprocessing.Process(target=monitor_gpu, args=(1, GPU_USAGE_FILE))
    p.start()

    train_start_time = time.time()
    logger.info(
        'Training for %d epochs, with batch size %d, vocabulary of %d and max sentence length %d...'
        % (epochs, batch_size, max_features, maxlen))
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test, y_test],
              callbacks=[logger_callback])

    p.terminate()

    report_dict['time'] = time.time() - train_start_time

    with open(EPOCH_STATS_LOGFILE, 'r') as f:
        report_dict['epoch_stats'] = f.readlines()

    with open(GPU_USAGE_FILE, 'r') as f:
        gpu_usages = [float(line.rstrip()) for line in f]
        report_dict['gpu_usage'] = np.mean(gpu_usages[1:])

    with open(REPORT_FILE, 'w') as f:
        f.write(json.dumps(report_dict, indent=4))

    logger.info('Reports generated!')
    logger.info('Finished running the benchmark!')
