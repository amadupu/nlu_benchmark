import tensorflow as tf
import spacy
import os
import json
import numpy as np
from paths import *

from utils import clean_dir
from utils import generate_data, get_file_timestamp


class TFEncoder(object):
    def __init__(self,builder):
        self.data_src = builder.data_src
        self.data_dest = builder.data_dest
        self.file_limit = builder.file_limit
        self.is_test_data = builder.is_test_data
        self.nlp = spacy.load('en_core_web_lg')


    def make_example(self,sent,label):
        ex = tf.train.SequenceExample()

        doc = self.nlp(sent)
        word_count = 0
        xs = list()
        for token in doc:
            if not token.is_punct and not token.is_stop and not token.is_space and token.has_vector:
                xs.append(token.vector)


        # add context features


        sequence_length = len(xs)
        # print('Generating Record of Length: {}'.format(sequence_length))
        ex.context.feature['len'].int64_list.value.append(sequence_length)
        ex.context.feature['label'].int64_list.value.append(label)

        # add sequence features
        fl_tokens = ex.feature_lists.feature_list['tokens']



        for token in xs:
            if np.ndim(token) == 0:
                fl_tokens.feature.add().float_list.value.append(token)
            else:
                fl_tokens.feature.add().float_list.value.extend(token)

        return ex



    def dump_records(self,data):


        data_len = len(data)
        count = 0


        if self.file_limit is None:
           limit = data_len
        else:
           limit = self.file_limit



        while count < data_len:
            batch = data[count : count + limit]
            target = os.path.join(self.data_dest, '{}.trf'.format(get_file_timestamp()))
            # print('Writing to {} record'.format(target))
            tf_writer = tf.python_io.TFRecordWriter(target)
            for sent,label in batch:
                ex = self.make_example(sent,label)
                tf_writer.write(ex.SerializeToString())


            count += limit


    # template
    def encode(self):

        try:


            # preprocess the data
            categories, data = generate_data(self.data_src,is_test=self.is_test_data)

            # make categories header
            with open('headers.json','w') as fp:
                json.dump(categories,fp)

            self.dump_records(data=data)
            return True

        except Exception as err:
            print(err)
            return False

    class Builder():

        def __init__(self):
            self.data_src = ''
            self.data_dest = ''
            self.file_limit = None
            self.is_test_data = False


        def set_data_source(self,val):
            self.data_src = val
            return self


        def set_data_dest(self,val):
            self.data_dest = val
            return self

        def set_file_limit(self,val):
            self.file_limit = val
            return self

        def set_test_data(self,flag):
            self.is_test_data = flag
            return self

        def build(self):
            return TFEncoder(self)


if __name__ == '__main__':

    clean_dir('records')

    encoder = TFEncoder.Builder().\
        set_data_dest(os.path.join('records','train')).\
        set_data_source(input_data_source_path).\
        set_file_limit(100).\
        set_test_data(False).\
        build()

    if encoder.encode() is True:
        print("Encode Train Successful")
    else:
        print('Encode Train Failure')

    encoder = TFEncoder.Builder().\
        set_data_dest(os.path.join('records','eval')).\
        set_data_source(input_data_source_path).\
        set_file_limit(100).\
        set_test_data(True).\
        build()


    if encoder.encode() is True:
        print("Encode Eval Successful")
    else:
        print('Encode Eval Failure')



