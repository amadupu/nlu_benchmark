
from data_decoder import TFDecoder
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    decoder = TFDecoder.Builder(). \
        set_feature_size(600). \
        set_num_epochs(1). \
        set_path('records/train'). \
        set_shuffle_status(True). \
        build()

    batch_input = tf.train.batch(tensors=decoder.dequeue(True), batch_size=100,
                                                  dynamic_pad=True,
                                                  allow_smaller_final_batch=True)




    # xs = batch_input[1]
    #
    # ys = batch_input[2]
    #
    # batch_size = tf.shape(xs)[0]


    # x_unpack = tf.unpack(xs,axis=1)


    l_init = tf.global_variables_initializer()
    g_init = tf.local_variables_initializer()

    # y_reshaped = tf.reshape(batch_input[2],[-1])

    with tf.Session() as sess:
        sess.run([l_init,g_init])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)
        try:
            while not coord.should_stop():
                result = sess.run(batch_input)

                print(np.shape(result[0]),np.shape(result[1]), result[2], result[3])
                # print(np.shape(result[4]),np.shape(result[5]))

        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()

        coord.join(threads=threads)

