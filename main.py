from eu_dataset import EuDataset
from utils import limit_mem, logger_setup
limit_mem()

from model import EuParlRnnModel
from argparse import ArgumentParser
import os
import numpy as np
import time
import pickle
import logging
from itertools import islice
import tensorflow as tf

#default arguments
data_folder = '/mnt/disk/datasets/europarl/clean_sentences/'
all_chars = pickle.load(open('pickles/all_chars_python27.pickle', 'rb'))
char2id = {c:i for i,c in enumerate(all_chars)}
id2char = {i:c for i,c in enumerate(all_chars)}

input_max_len = 200
rnn_hidden_size = 500
embedding_len = 200
num_embeddings = len(all_chars)
dropout_rate = 0.5
learning_rate = 1e-4
num_classes = 21
batch_size = 64
train_epochs = 5

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--output-folder', required=True,
                        dest='output_folder', default='output')
    parser.add_argument('--validate', dest='validate', default='no')
    parser.add_argument('--saved-model', dest='saved_model', default='')
    parser.add_argument('--data-folder', dest='data_folder', default=data_folder)
    parser.add_argument('--batch-size', dest='batch_size', default=batch_size)
    return parser

def mini_validation_accuracy(dataset_instance, session, model_input,
                             labels, is_train, acc, loss):
    y_batch = labels
    calculate_on_batches = 200
    acc_list = []
    loss_list = []
    for (val_batch, val_labels) in islice(dataset_instance.get_epoch_data('val'),
                                          0, calculate_on_batches):
        feed_dict = {model_input: val_batch, y_batch: val_labels, is_train: False}
        val_acc, val_loss = session.run([acc, loss], feed_dict=feed_dict)
        acc_list.append(val_acc)
        loss_list.append(val_loss)
    return np.average(acc_list), np.average(loss_list)

def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists(options.output_folder):
        os.makedirs(options.output_folder)
    logging = logger_setup(output_folder=options.output_folder)

    eu_dataset = EuDataset(batch_size=int(options.batch_size),
                           data_folder=options.data_folder,
                           input_max_len=input_max_len,
                           char2id=char2id,
                           debug=False)

    if not os.path.exists('models'):
        os.makedirs('models')

    start_time = time.strftime("%h%d_%H_%M")
    model_config = {
        'input_max_len': input_max_len,
        'num_embeddings': num_embeddings,
        'embedding_len': embedding_len,
        'rnn_hidden_size': rnn_hidden_size,
        'dropout_rate': dropout_rate,
        'num_classes': num_classes,
        'learning_rate': learning_rate,
        'train_epochs': train_epochs
    }
    model = EuParlRnnModel(model_config=model_config, logging=logging)
    model.build_model()

    if options.validate == 'yes':
        print('Final model evaluation ...')
        model_path = options.saved_model
        model.final_validation(dataset_instance=eu_dataset,
                               model_path=model_path)
    else:
        print('Training ...')
        model.train(eu_dataset)

if __name__ == '__main__':
    main()
