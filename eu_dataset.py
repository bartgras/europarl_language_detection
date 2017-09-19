import numpy as np
from keras.preprocessing.sequence import pad_sequences

class EuDataset:
    def __init__(self, batch_size, data_folder, input_max_len, char2id, debug=False):
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.input_max_len = input_max_len
        self.char2id = char2id
        self.debug = debug
        self.lang_codes = ['sv','es','et','el','it','fr', \
                           'fi','sk','ro','cs','de','nl', \
                           'pt','hu','pl','lt','lv','bg', \
                           'da','sl','en']
        self.num_classes = len(self.lang_codes)
        self.lang_class = {code:i for i,code in enumerate(self.lang_codes)}

    def lang_classes(self):
        return self.lang_class

    def file_path(self, ds_type=None):
        """
        filename depends on dataset type (str): train/text/val
        if debug is True it allows to test code on fraction of code,
        example: {path_to_data_folder/}all_test_shuffled_debug.txt
        """
        if self.debug:
            debug_str = '_debug'
        else:
            debug_str = ''

        return '%sall_%s_data_shuffled%s.txt' % (self.data_folder,                                                                                ds_type, # train/test/val
                                                 debug_str)

    def get_epoch_data(self, ds_type='train'):
        position_counter = 0
        if ds_type == 'final_test':
            filepath = self.data_folder + 'europarl.test'
            split_chars = "\t"
        else:
            filepath = self.file_path(ds_type)
            split_chars = "||"

        with open(filepath) as f:
            print("path: %s, counter: %s" % (filepath, position_counter))
            batch_ids = []
            batch_labels = []
            for line in f:
                lang_code, sentence = line.split(split_chars)
                position_counter += 1

                # TODO: speed up this part
                #batch_ids.append([self.char2id.get(char) for char in sentence.decode('utf-8')])
                sentence_ids = []
                for char in sentence.decode('utf-8'):
                    char_id = self.char2id.get(char)
                    if char_id is not None:
                        sentence_ids.append(char_id)
                batch_ids.append(sentence_ids)

                batch_labels.append(self.lang_class[lang_code])
                if position_counter % self.batch_size == 0:
                    raw_batch = pad_sequences(batch_ids, 
                                              self.input_max_len)
                    batch_data = np.array(raw_batch)
                    yield (batch_data,
                           np.array(batch_labels))
                    batch_ids = []
                    batch_labels = []

    def __steps_per_epoch(self, filename):
        return int(sum((1 for _ in open(filename))) / self.batch_size)

    def train_steps_per_epoch(self):
        return self.__steps_per_epoch(self.file_path('train'))

    def validation_steps_per_epoch(self):
        return self.__steps_per_epoch(self.file_path('val'))
