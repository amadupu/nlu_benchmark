import os
import tensorflow as tf
import numpy as np



class TFDecoder(object):
    def __init__(self,builder):
        self.num_epochs = builder.num_epochs
        self.path = builder.path
        self.shuffle = builder.shuffle
        self.feature_size = builder.feature_size
        self.label_size = builder.label_size


        filelist = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        self.fqueue = tf.train.string_input_producer(filelist,
                                        shuffle=self.shuffle,
                                        seed=None,
                                        num_epochs=self.num_epochs)

    def dequeue(self,classifer=False):
        reader = tf.TFRecordReader()
        key, ex = reader.read(self.fqueue)
        context_features = {
            "len": tf.FixedLenFeature([], dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([self.feature_size], dtype=tf.float32),
            "entity_labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=ex,
                                                                           context_features=context_features,
                                                                           sequence_features=sequence_features)
        # return context_parsed["len"], context_parsed["id"] ,  context_parsed["seq-id"],  context_parsed["label"], sequence_parsed["tokens"], sequence_parsed["labels"]

        # if classifer:
        #     return context_parsed["len"], sequence_parsed["tokens"], context_parsed["label"]
        # else:
        return context_parsed["len"], sequence_parsed["tokens"], sequence_parsed["entity_labels"], context_parsed["label"]

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
        set_feature_size(600).\
        set_num_epochs(1).\
        set_path(os.path.join('records','train')).\
        set_shuffle_status(True).\
        build()

    batch_input = decoder.dequeue(True)

    l_init = tf.global_variables_initializer()
    g_init = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run([l_init,g_init])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            while not coord.should_stop():
                result = sess.run(batch_input)
                print(result[0],np.shape(result[1]),result[2] , result[3])
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()

        coord.join(threads=threads)




