from keras import backend as K
import tensorflow as tf
import os
import time
import logging

def logger_setup(output_folder='output'):
    logging.basicConfig(filename='%s/training.log' % output_folder,
                        filemode='a',
                        format='%(asctime)s, %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    return logging

def limit_mem():
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
                            device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    K.set_session(K.tf.Session(config=config))
