import tensorflow as tf
from data_decoder import TFDecoder
from enum import Enum
import numpy as np
import os

class RNNModel(object):
    def __init__(self,builder):

        self.epochs = builder.epochs
        # batch size
        self.batch_size = builder.batch_size
        self.read_path = builder.read_path
        self.feature_size = builder.feature_size
        self.num_layers = builder.num_layers
        self.max_steps = builder.max_steps
        self.num_classes = builder.num_classes
        self.cell_size = builder.cell_size
        self.cell_type = builder.cell_type
        self.learning_rate = builder.learning_rate
        self.threads= None
        self.batch_inputs = None
        self.evalfn = builder.evalfn
        self.oper_mode = builder.oper_mode
        self.global_step = None
        self.validation_step = builder.validation_step
        self.logs_path = builder.logs_path
        self.model_path = builder.model_path
        self.model_name = builder.model_name
        self.is_classifier = builder.is_classifier
        self.is_timemajor = builder.is_timemajor
        self.is_bidirectional = builder.is_bidirectional
        self.is_state_feedback = builder.is_state_feedback
        self.build_graph()

    def build_graph(self):
        with tf.Graph().as_default():

            # if self.oper_mode == RNNModel.OperMode.TEST or \
            #         self.oper_mode == RNNModel.OperMode.OPER_MODE_EVAL:
            #     filename = ".".join([tf.train.latest_checkpoint(self.model_path),'meta'])
            #     self.saver = tf.train.import_meta_graph(filename)
            # else:
                with tf.name_scope('input_pipe_line'):
                    if self.oper_mode == RNNModel.OperMode.OPER_MODE_TEST:
                        self.xs = tf.placeholder(tf.float32,[1,None,self.feature_size],name='xs')
                        # pad's second argument can be seen as [[up, down], [left, right]]
                        if self.is_classifier:
                            self.ys = tf.placeholder(tf.int64, [None], name='ys')
                        else:
                            self.ys = tf.placeholder(tf.int64, [1, None], name='ys')

                        self.steps = tf.placeholder(tf.int64, [None], name='steps')
                    else:
                        decoder = TFDecoder.Builder(). \
                            set_feature_size(self.feature_size). \
                            set_num_epochs(self.epochs). \
                            set_path(self.read_path). \
                            set_shuffle_status(True). \
                            build()
                        self.steps, self.xs , self.ys = tf.train.batch(tensors=decoder.dequeue(self.is_classifier), batch_size=self.batch_size,
                                                                  dynamic_pad=True,
                                                                  allow_smaller_final_batch=True,name='batch_processor')
                        self.global_step = tf.Variable(0, name="global_step", trainable=False)


                xs = self.xs
                ys = self.ys

                # if self.is_timemajor is True:
                #     xs = tf.transpose(self.xs,perm=[1,0,2])
                #     ys = tf.transpose(self.ys)
                # else:
                #     xs = self.xs
                #     ys = self.ys

                self.keepprob = tf.placeholder(tf.float32, [], name='keeprob')

                # input weights
                with tf.name_scope('input_layer'):
                    with tf.name_scope('Weigths'):
                        Win = self.weight_variable([self.feature_size,self.cell_size],name='W_in')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('input_layer/Weights',Win)
                    with tf.name_scope('Biases'):
                        Bin = self.bias_variable([self.cell_size], name='B_in')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('input_layer/Biases', Bin)


                    x = tf.reshape(xs,[-1,self.feature_size])


                    rnn_inputs = tf.add(tf.matmul(x,Win),Bin)
                    # self.input_test = rnn_inputs

                    rnn_inputs = tf.nn.dropout(rnn_inputs,keep_prob=self.keepprob)

                    self.rnn_inputs = tf.reshape(rnn_inputs, [-1, tf.shape(self.xs)[1], self.cell_size],
                                                 name='rnn_inputs')

                    # if self.is_timemajor is True:
                    #     self.rnn_inputs = tf.reshape(rnn_inputs,[tf.shape(self.xs)[1],-1,self.cell_size],name='rnn_inputs')
                    # else:
                    #     self.rnn_inputs = tf.reshape(rnn_inputs, [-1, tf.shape(self.xs)[1], self.cell_size],
                    #                                  name='rnn_inputs')

                with tf.name_scope('rnn_layer'):
                    if self.cell_type == RNNModel.CellType.RNN_CEL_TYPE_LSTM:
                        cell_fw = tf.nn.rnn_cell.LSTMCell(self.cell_size, state_is_tuple=True,name='LSTMCell')
                        cell_bw = tf.nn.rnn_cell.LSTMCell(self.cell_size, state_is_tuple=True, name='LSTMCell')
                    else:
                        cell_fw = tf.nn.rnn_cell.GRUCell(self.cell_size,name='GRUCell')
                        cell_bw = tf.nn.rnn_cell.GRUCell(self.cell_size, name='GRUCell')

                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,input_keep_prob=self.keepprob)
                    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=self.keepprob)

                    if self.cell_type == RNNModel.CellType.RNN_CEL_TYPE_LSTM:
                        cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                        cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)
                    else:
                        cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * self.num_layers)
                        cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * self.num_layers)

                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,output_keep_prob=self.keepprob)
                    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keepprob)

                    batch_size = tf.shape(self.xs)[0]

                    # self.tf_batch_size = batch_size

                    self.state_fw = cell_fw.zero_state(batch_size, tf.float32)
                    self.state_bw = cell_bw.zero_state(batch_size, tf.float32)


                    if self.is_bidirectional is True:

                        outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                                         cell_bw=cell_bw,
                                                                                         inputs=self.rnn_inputs,
                                                                                         sequence_length=self.steps,
                                                                                         initial_state_fw=self.state_fw,
                                                                                         initial_state_bw=self.state_bw,
                                                                                         time_major=False )




                        rnn_outputs = tf.concat(outputs, 2)
                        cell_size = self.cell_size * 2
                        # (self.state_fw, self.state_bw) = self.final_state
                    else:

                        outputs, self.final_state = tf.nn.dynamic_rnn(cell_fw,self.rnn_inputs,sequence_length=self.steps,
                                                                      initial_state=self.state_fw,time_major=False)


                        rnn_outputs = outputs
                        cell_size = self.cell_size
                        # self.state_fw = self.final_state


                    self.check_tensor_1 = outputs
                    self.check_tensor_2 = self.final_state
                    self.check_tensor_3 = rnn_outputs





                    # self.output_concat = rnn_outputs

                    # if time is major we want the state to be fed back
                    # if self.is_timemajor:


                    # (self.state_fw, self.state_bw) = self.final_state


                    # rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell_fw,self.rnn_inputs,sequence_length=self.steps,
                    #                                                   initial_state=self.state_fw,time_major=False)


                    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob=self.keepprob,name='rnn_outputs')


                    #
                    # rnn_outputs = tf.transpose(rnn_outputs,perm=[1,0,2])

                with tf.name_scope('output_layer'):
                    with tf.name_scope('Weights'):
                        Wout = self.weight_variable([cell_size,self.num_classes],name='W_out')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('output_layer/Weights',Wout)
                    with tf.name_scope('Biases'):
                        Bout = self.bias_variable([self.num_classes],name='B_out')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('outpu_layer/Weights',Bout)


                    if self.is_classifier is True:
                        # get last rnn output
                        # self.test_rnn_outputs = tf.transpose(rnn_outputs,perm=[1,0,2])[-1]

                        # if self.is_timemajor is True:
                        #     rnn_outputs = rnn_outputs[-1]
                        # else:
                        self.idx0 = tf.range(tf.cast(batch_size,tf.int64))
                        self.idx1 = self.idx0 * tf.cast(tf.shape(tf.cast(rnn_outputs,tf.int64))[1],tf.int64)
                        self.idx2 =  self.idx1 + (self.steps - 1)
                        rnn_outputs = tf.gather(tf.reshape(rnn_outputs, [-1, cell_size]), self.idx2)

                    else:
                        rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size])


                    #for seq2seq calcuations
                    self.rnn_outputs = rnn_outputs

                    logits = tf.add(tf.matmul(rnn_outputs,Wout),Bout)

                    # for seq2seq calculations
                    self.logits = tf.nn.softmax(logits)

                    self.predictions = tf.argmax(self.logits,axis=-1)
                    self.flat_labels = tf.reshape(ys, [-1])

                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.flat_labels,self.predictions),tf.float32),name='accuracy')
                    tf.summary.scalar('Accuracy',self.accuracy)

                with tf.name_scope('cross_entropy'):
                    self.loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.flat_labels))
                    # if self.is_classifier:
                    #     self.loss = tf.reduce_mean(
                    #         tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.flat_labels))
                    #
                    # else:
                    #     if self.is_timemajor is True:
                    #         self.seq_logits =  tf.unstack(tf.reshape(logits, [tf.shape(self.xs)[1], -1, self.num_classes]),axis=0)
                    #         self.seq_lables =  tf.unstack(ys, num=self.max_steps, axis=0)
                    #         self.seq_weights = tf.unstack(tf.sign(tf.reduce_max(tf.abs(xs), axis=-1)),num=self.max_steps, axis=0)
                    #         self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.seq_logits, self.seq_lables,
                    #                                                             self.seq_weights, name='loss')
                    #     else:
                    #         self.loss = tf.reduce_mean(
                    #             tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.flat_labels))

                    tf.summary.scalar('Cross Entropy', self.loss)
                    # weights = tf.reshape
                    # tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits=logits, labels=self.logits)
                if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                    with tf.name_scope('train'):
                        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step,name='train_step')

                self.saver = tf.train.Saver(tf.global_variables(),keep_checkpoint_every_n_hours=1,max_to_keep=2)

                self.g_init = tf.global_variables_initializer()
                self.l_init = tf.local_variables_initializer()



                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

                self.summary = tf.summary.merge_all()

                if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                    self.summary_writer = tf.summary.FileWriter(self.logs_path + '/train' ,self.sess.graph)
                elif self.oper_mode == RNNModel.OperMode.OPER_MODE_EVAL:
                    self.summary_writer = tf.summary.FileWriter(self.logs_path + '/eval', self.sess.graph)


    def weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def reset_graph(self):
        if self.sess is not None:
            self.sess.close()
        tf.reset_default_graph()

    def init_graph(self):


        try:
            print('Restoring Lastest Checkpoint : ',end=' ')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
            print('SUCCESS')


        except Exception as err:
            print('FAILED')
            self.sess.run(self.g_init)

        finally:
            self.sess.run(self.l_init)
            self.coord = tf.train.Coordinator()

    def train(self,keepprob=1.0):

        self.init_graph()

        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        path = os.path.join(self.model_path, self.model_name)

        try:
            feed_dict = {self.keepprob : keepprob}

            count = 0
            total_loss = 0.0
            total_acc = 0.0

            while not self.coord.should_stop():


                # c1,c2,c3,pred, logits, fw,bw,fs,xs,ys,yf,steps  = self.sess.run([self.check_tensor_1, self.check_tensor_2,self.check_tensor_3,self.predictions, self.logits, self.state_fw, self.state_bw, self.final_state, self.xs, self.ys, self.flat_labels, self.steps],feed_dict)
                # print('c1: ',np.shape(c1))
                # print('c2: ',np.shape(c2))
                # print('c3: ', np.shape(c3))
                # print('fw: ',np.shape(fw))
                # print('bw: ',np.shape(bw))
                # print('fs: ',np.shape(fs))
                #
                # print('pred: ', np.shape(pred))
                # print('logits: ',np.shape(logits))
                #
                # print('xs: ',np.shape(xs))
                # print('ys: ',ys, np.shape(ys))
                # print('yf: ',yf, np.shape(yf))
                # print('steps: ',steps, np.shape(steps))
                # continue

                _,loss,  accuracy, summary, final_state, state_fw, ys, steps, pred     = self.sess.run([self.train_step,self.loss, self.accuracy, self.summary, self.final_state, self.state_fw, self.flat_labels, self.steps, self.predictions],feed_dict)
                total_loss += loss
                total_acc += accuracy
                count += 1

                # print('Accuracy: {} final_state: {}'.format(accuracy,np.shape(final_state)))
                # print('init_state: ', state_fw)
                # print('ys: ',ys)
                # print('pred: ',pred)
                # print('steps: ',steps)

                #_, loss, accuracy, summary, rnn_outputs, logits, predictions, flat_labels = self.sess.run(
                #    [self.train_step, self.loss, self.accuracy, self.summary, self.rnn_outputs, self.logits, self.predictions, self.flat_labels], feed_dict)
                # print('State:{} {} {}'.format(self.oper_mode,np.shape(state),state))
                # print(final_state)
                # print('Final State:{} {} {}'.format(self.oper_mode,np.shape(final_state), final_state))

                current_step = tf.train.global_step(self.sess, self.global_step)
                print('Train: ',current_step,loss,accuracy)
                if current_step % self.validation_step == 0:
                    self.summary_writer.add_summary(summary, current_step)
                    self.summary_writer.flush()
                    # print(seq_weights)
                    # print('Saving model params for step: ', current_step)

                    print('{} Loss: {} Accuarcy {}'.format(self.oper_mode, total_loss / count, total_acc / count))
                    total_loss = 0.0
                    total_acc = 0.0
                    count = 0

                    self.saver.save(self.sess, path, global_step=current_step, write_meta_graph=False)
                    if self.evalfn:
                        self.evalfn(current_step)



                # print(np.shape(pred), np.shape(labels), accuracy)
                # print(pred,labels)
                if self.is_state_feedback is True:
                    if self.is_bidirectional is True:
                        (state_fw,state_bw) = final_state
                        feed_dict[self.state_fw] = state_fw
                        feed_dict[self.state_bw] = state_bw
                    else:
                        feed_dict[self.state_fw] = final_state




        except tf.errors.OutOfRangeError:
            print('Out Of Range Errror')
            # pass
        finally:
            self.summary_writer.close()
            self.coord.request_stop()

        current_step = tf.train.global_step(self.sess, self.global_step)
        self.saver.save(self.sess, path, global_step=current_step,write_meta_graph=False)

        self.coord.join(self.threads)
        self.threads = None



    def evaluate(self, curr_step, keepprob=1.0):

        self.init_graph()

        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        count = 0
        total_loss = 0.0
        total_acc = 0.0
        summary = None
        try:
            feed_dict = {self.keepprob: keepprob}

            while not self.coord.should_stop():
                loss, accuracy, summary = self.sess.run([self.loss,self.accuracy, self.summary],feed_dict)
                total_loss += loss
                total_acc += accuracy
                count += 1




        except Exception as err:
            print('{} Out of Range'.format(self.oper_mode))
        finally:
            self.coord.request_stop()
            if count > 0:
                self.summary_writer.add_summary(summary, curr_step)
                self.summary_writer.flush()
                print('{} Loss : {}  Accuracy: {}'.format(self.oper_mode, total_loss/count, total_acc/count))
        self.coord.join(self.threads)
        self.threads = None

    def test(self,xs,steps):
        feed_dict = {self.keepprob: 1.0, self.xs : xs, self.steps : steps}

        result = self.sess.run(self.predictions, feed_dict)

        # print(score[:steps[0]])

        # print(score,steps)

        # logits = np.sum(logits,axis=0)/steps
        #
        # if not self.is_classifier:
        #     score = np.sum(score)/steps

        return result


    class Builder():
        def __init__(self):
            self.epochs = 1
            self.batch_size = 20
            self.read_path = ''
            self.feature_size = 300
            self.num_classes = 8
            self.cell_size = 128
            self.max_steps = 50
            self.num_layers = 1
            self.learning_rate = 1e-4
            self.logs_path = ''
            self.model_path = ''
            self.model_name = ''
            self.cell_type = RNNModel.CellType.RNN_CEL_TYPE_LSTM
            self.evalfn = None
            self.oper_mode = RNNModel.OperMode.OPER_MODE_NONE
            self.validation_step = 10
            self.is_classifier = False
            self.is_timemajor = False
            self.is_bidirectional = False
            self.is_state_feedback = False

        def set_state_feedback(self,flag):
            self.is_state_feedback = flag
            return self


        def set_bi_directional(self,flag):
            self.is_bidirectional = flag
            return self

        def set_epochs(self,val):
            self.epochs = val
            return self

        def set_batch_size(self,val):
            self.batch_size = val
            return self

        def set_read_path(self,val):
            self.read_path = val
            return self

        def set_feature_size(self,val):
            self.feature_size = val
            return self

        def set_class_size(self,val):
            self.num_classes = val
            return self

        def set_cell_size(self,val):
            self.cell_size = val
            return self

        def set_cell_type(self,val):
            self.cell_type = val
            return self

        def set_max_steps(self,val):
            self.max_steps = val
            return self

        def set_layer_size(self,val):
            self.num_layers = val
            return self

        def set_learning_rate(self,val):
            self.learning_rate = val
            return self

        def set_eval_fn(self,val):
            self.evalfn = val
            return self

        def set_oper_mode(self,val):
            self.oper_mode = val
            return self

        def set_validation_step(self,val):
            self.validation_step = val
            return self

        def set_logs_path(self,val):
            self.logs_path = val
            return self

        def set_model_path(self,val):
            self.model_path = val
            return self

        def set_model_name(self,val):
            self.model_name = val
            return self

        def set_classifer_status(self,flag):
            self.is_classifier = flag
            return self

        def set_time_major(self,flag):
            self.is_timemajor = flag
            return self

        def build(self):
            return RNNModel(builder=self)


    class CellType(Enum):
        RNN_CELL_TYPE_NONE = 0
        RNN_CEL_TYPE_LSTM = 1
        RNN_CELL_TYPE_GRU = 2

    class OperMode(Enum):
        OPER_MODE_NONE = 0
        OPER_MODE_TRAIN = 1
        OPER_MODE_EVAL = 2
        OPER_MODE_TEST = 3




