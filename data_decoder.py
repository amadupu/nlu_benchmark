import os
import tensorflow as tf
import numpy as np
from paths import *



class TFDecoder(object):
    def __init__(self,builder):
        self.num_epochs = builder.num_epochs
        self.path = builder.path
        self.shuffle = builder.shuffle
        self.feature_size = builder.feature_size
        self.label_size = builder.label_size


        filelist = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        # print(filelist.sort())
        self.fqueue = tf.train.string_input_producer(filelist,
                                        shuffle=self.shuffle,
                                        seed=None,
                                        num_epochs=self.num_epochs)




    def dequeue(self,classifer=False):
        reader = tf.TFRecordReader()
        key, ex = reader.read(self.fqueue)
        context_features = {
            "len": tf.FixedLenFeature([], dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64),
            "id": tf.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([self.feature_size], dtype=tf.float32),
            "entity_labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "char_tokens": tf.VarLenFeature(dtype=tf.int64),
            "char_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),

        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=ex,
                                                                           context_features=context_features,
                                                                           sequence_features=sequence_features)
        # return context_parsed["len"], context_parsed["id"] ,  context_parsed["seq-id"],  context_parsed["label"], sequence_parsed["tokens"], sequence_parsed["labels"]

        # if classifer:
        #     return context_parsed["len"], sequence_parsed["tokens"], context_parsed["label"]
        # else:
        return context_parsed["len"], sequence_parsed["tokens"], sequence_parsed["entity_labels"], sequence_parsed["char_tokens"], sequence_parsed["char_len"], context_parsed["label"],

    class Builder():
        def __init__(self):
            self.num_epochs = 1
            self.path = ''
            self.shuffle=False
            self.feature_size = None
            self.label_size = None


        def set_num_epochs(self,val):
            self.num_epochs = val
            return self

        def set_path(self,val):
            self.path = val
            return self

        def set_shuffle_status(self,flag):
            self.shuffle = flag
            return self

        def set_feature_size(self,val):
            self.feature_size = val
            return self

        def set_label_size(self,val):
            self.label_size = val
            return self

        def build(self):
            return TFDecoder(self)


if __name__ == '__main__':

    decoder = TFDecoder.Builder().\
        set_feature_size(word_feature_size + pos_feature_size).\
        set_num_epochs(1).\
        set_path(os.path.join('records','train')).\
        set_shuffle_status(False).\
        build()

    batch_input = decoder.dequeue(True)



    dense = tf.sparse_tensor_to_dense(batch_input[3])

    embeddings = tf.get_variable(
        name='word_embeddings',
        dtype=tf.float32,

        shape=[char_vocab_size, char_feature_size])

    word_vec = tf.nn.embedding_lookup(embeddings, dense)

    l_init = tf.global_variables_initializer()
    g_init = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run([l_init,g_init])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            while not coord.should_stop():
                result ,dense_char, char_vec = sess.run([batch_input,dense,word_vec])
                print(result[0],np.shape(result[1]), np.shape(char_vec), np.shape(dense_char))
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()

        coord.join(threads=threads)




