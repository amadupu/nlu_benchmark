import tensorflow as tf
import spacy
import os
import json
import numpy as np
from paths import *

from utils import *



class TFEncoder(object):
    def __init__(self,builder):
        self.data_src = builder.data_src
        self.data_dest = builder.data_dest
        self.file_limit = builder.file_limit
        self.is_test_data = builder.is_test_data
        self.nlp = spacy.load(spacy_model_path)
        self.nlp_pos = spacy.load('en_core_web_sm')
        self.id = 1
        self.pos_map = dict()

        # build char vocab
        build_char_vocab(input_data_source_path)
        with open('vocab_char.txt') as fp:
            vocab = json.load(fp)

        self.vocab = vocab['vocab']



    def make_example(self,sent,label):
        ex = tf.train.SequenceExample()
        phrase = r''
        entity_labels = list()
        word_vec = list()
        # word_vec_len = list()
        sent_len = len(sent)
        for index,item in enumerate(sent):
            phrase += item[0]
            if index < sent_len - 1:
                phrase += ' '
            entity_labels.append(item[1])
            char_vec = list()
            for ch in item[0]:
                try:
                    char_vec.append(self.vocab.index(ch))

                except Exception as err:
                    print('ERROR: Unable to find ch {} in vocab'.format(ch))
            word_vec.append(char_vec)
            # word_vec_len.append(len(char_vec))

        phrase.lower()
        doc = self.nlp_pos(phrase)
        if len(doc) != len(sent):
            print('\nDoc: ',end='')
            for tok in doc:
                print(tok.text_with_ws, end='')
            print('\nSent: ', phrase)
            raise Exception('Doc Sent Size Mismatch')

        word_count = 0
        xs = list()

        for token in doc:
            pos = token.pos_


            if pos == 'CCONJ':
                pos = 'CONJ'
            elif pos == 'PROPN':
                pos = 'PROP'
            elif pos == 'PUNCT':
                pos = 'PUN'
            elif pos == 'SCONJ':
                pos = 'SCO'

            if not self.nlp(pos)[0].has_vector:
                   raise Exception('Invalid vector for pos: ', pos )

            pos_vector   = self.nlp(pos)[0].vector[:pos_feature_size]
            token = self.nlp(token.text)[0]

            if not token.has_vector:
                token_vector = self.nlp('UNK')[0].vector
            else:
                token_vector = token.vector

            token_vector = token_vector[:word_feature_size]

            if pos_feature_size > 0:
                xs.append(np.concatenate((token_vector,pos_vector),axis=-1))
            else:
                xs.append(token_vector)

            # print(np.shape(xs),pos,np.shape(pos_vector))
            # xs.append(token_vector)
            # if not token.is_punct and not token.is_stop and not token.is_space and token.has_vector:



        # add context features


        sequence_length = len(xs)

        if sequence_length != len(entity_labels):
            raise Exception('Input Vector and Entity Label Size mismatch', sequence_length, len(entity_labels))
        # print('Generating Record of Length: {}'.format(sequence_length))
        ex.context.feature['len'].int64_list.value.append(sequence_length)
        ex.context.feature['label'].int64_list.value.append(label)
        ex.context.feature['id'].int64_list.value.append(self.id)
        self.id += 1

        # add sequence features
        fl_tokens = ex.feature_lists.feature_list['tokens']
        fl_labels = ex.feature_lists.feature_list['entity_labels']
        fl_chars = ex.feature_lists.feature_list['char_tokens']
        fl_char_len = ex.feature_lists.feature_list['char_len']



        for token,ent_label,char_vec in zip(xs,entity_labels,word_vec):
            if np.ndim(token) == 0:
                fl_tokens.feature.add().float_list.value.append(token)
            else:
                fl_tokens.feature.add().float_list.value.extend(token)

            fl_chars.feature.add().int64_list.value.extend(char_vec)

            fl_labels.feature.add().int64_list.value.append(ent_label)
            fl_char_len.feature.add().int64_list.value.append(len(char_vec))

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
            print('Writing to {} record'.format(target))
            tf_writer = tf.python_io.TFRecordWriter(target)
            for sent,label in batch:
                ex = self.make_example(sent,label)
                tf_writer.write(ex.SerializeToString())


            count += limit


    # template
    def encode(self):

        try:


            # preprocess the data
            headers,  data = generate_data(self.data_src,is_test=self.is_test_data)

            # capture headers only for trainng data
            if not self.is_test_data:
                # make categories header
                with open('headers.json','w') as fp:
                    json.dump(headers,fp)


            print('dump records')
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
         set_data_dest(os.path.join('records','eval')).\
         set_data_source(input_data_source_path).\
         set_file_limit(100).\
         set_test_data(True).\
         build()

    if encoder.encode() is True:
         print("Encode Eval Successful")
    else:
         print('Encode Eval Failure')
    
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





