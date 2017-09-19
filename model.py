import tensorflow as tf
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
import numpy as np
from itertools import islice

def fully_connected(previous_layer_output_matrix, hidden_layer_size, name):
    previous_layer_size = previous_layer_output_matrix.get_shape()[-1]
    with tf.variable_scope('%s_layer' % name):
        hidden_W = tf.get_variable('%s_layer_W' % name, 
                                   shape=[previous_layer_size, hidden_layer_size],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        hidden_b = tf.get_variable('%s_layer_b' % name, 
                                   shape=[hidden_layer_size],
                                   initializer=tf.constant_initializer(0.1))
        return tf.add(tf.matmul(previous_layer_output_matrix, 
                               hidden_W, name='%s_matmul' % name), 
                      hidden_b) 

class EuParlRnnModel:
    def __init__(self, model_config, logging):
        self.graph = tf.Graph()
        self.c = model_config
        self.logging = logging

    def build_model(self):
        tf.reset_default_graph()
        with self.graph.as_default():
            self.model_input = tf.placeholder(tf.int32, shape=(None,
                                                                 self.c['input_max_len']))
            self.y_batch = tf.placeholder(tf.int64, shape=(None,))
            self.is_train = tf.placeholder(tf.bool, None)

            embedding = Embedding(self.c['num_embeddings'],
                                  output_dim=self.c['embedding_len'],
                                  mask_zero=True)(self.model_input)
            rnn = Bidirectional(LSTM(self.c['rnn_hidden_size']))(embedding)
            if self.is_train is not None:
                rnn = tf.nn.dropout(rnn, self.c['dropout_rate'])
            output = Dense(self.c['num_classes'])(rnn)

            # Alternative "pure" Tensorflow based implementation
            # embedding = tf.get_variable('embedding', shape=[self.c['num_embeddings'],
            #                                                 self.c['embedding_len']],
            #                             initializer=tf.random_uniform_initializer(-0.1, 0.1))
            # embedding_lookup = tf.nn.embedding_lookup(embedding, self.model_input)

            # with tf.variable_scope('Recurrent'):
            #     rnn = tf.contrib.rnn.LSTMCell(self.c['rnn_hidden_size']) ## initializer!!!
            #     rnn_output, _ = tf.nn.dynamic_rnn(rnn, embedding_lookup, dtype=tf.float32)
            # output = fully_connected(rnn_output[:,-1,:], self.c['num_classes'],
            #                          name='model_output')

            self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.y_batch,
                                                                              output))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.c['learning_rate'])
            self.train_step = self.optimizer.minimize(self.loss)

            self.acc = tf.contrib.metrics.accuracy(tf.argmax(output, axis=1),
                                                   self.y_batch)
            self.model_saver = tf.train.Saver()

    def train(self, dataset_instance):
        tf.reset_default_graph()
        with tf.Session(graph=self.graph) as session:
            session.run(tf.global_variables_initializer())
            global_step = 0
            for epoch in range(self.c['train_epochs']):
                epoch_txt = 'Epoch: %s, batches in epoch: %s'
                steps_per_epoch = dataset_instance.train_steps_per_epoch()
                self.logging.info(epoch_txt % (epoch, steps_per_epoch))
                train_epoch_data = dataset_instance.get_epoch_data('train')
                for (X_batch, labels) in train_epoch_data:
                    feed_dict = {self.model_input: X_batch,
                                 self.y_batch: labels,
                                 self.is_train: True}
                    ts, batch_loss, batch_acc = session.run([self.train_step,
                                                             self.loss, self.acc],
                                                            feed_dict=feed_dict)
                    global_step += 1
                    if global_step % 100 == 0:
                        info_txt = "Batch: %s/%s, Batch train loss: %s, batch train accuracy: %s"
                        print(info_txt % (global_step, steps_per_epoch,
                                          batch_loss, batch_acc))
                        self.logging.info(info_txt % (global_step, steps_per_epoch,
                                                 batch_loss, batch_acc))

                    # Check accuracy on small sample of validation set
                    if global_step % 3000 == 0:
                        self.calc_validation_accuracy(session, dataset_instance)

                    # Save model
                    if global_step % 10000 == 0:
                        val_loss, val_acc = self.calc_validation_accuracy(session,
                                                                          dataset_instance)
                        self.save_model(session, global_step, val_acc, val_loss)

    def final_validation(self, dataset_instance, model_path):
        with tf.Session(graph=self.graph) as session:
            self.model_saver.restore(session, model_path)
            validation_batches = dataset_instance.validation_steps_per_epoch()
            print('Performing final validation on %s batches' % validation_batches)
            val_loss, val_acc = \
                self.calc_validation_accuracy(session=session,
                                              dataset_instance=dataset_instance,
                                              calculate_on_batches=validation_batches)
            val_acc_perc = val_acc * 100
            print('Final validation loss: %s, accuracy: %.3f%%' % (val_loss, val_acc_perc))


    def save_model(self, session, global_step, val_acc, val_loss):
        model_save_txt = 'models/model_step_%s_val_acc_%s' 
        self.model_saver.save(session, model_save_txt % (global_step, val_acc))
        log_save_txt = 'Saving model. Val loss: %s, val acc: %s'
        self.logging.info(log_save_txt % (val_loss, val_acc))

    def calc_validation_accuracy(self, session, dataset_instance, calculate_on_batches=200):
        epoch_data = dataset_instance.get_epoch_data('val')

        acc_list = []
        loss_list = []
        for (val_batch, val_labels) in islice(epoch_data, 0, calculate_on_batches):
            feed_dict = {self.model_input: val_batch,
                         self.y_batch: val_labels,
                         self.is_train: False}
            val_acc, val_loss = session.run([self.acc, self.loss], feed_dict=feed_dict)
            acc_list.append(val_acc)
            loss_list.append(val_loss)
        log_save_txt = 'Validation checkpoint. Loss: %s, val acc: %s'
        avg_loss = np.average(loss_list)
        avg_acc = np.average(acc_list)
        self.logging.info(log_save_txt % (avg_loss, avg_acc))
        return avg_loss, avg_acc
