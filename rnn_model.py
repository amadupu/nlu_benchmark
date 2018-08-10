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
        self.word_feature_size = builder.word_feature_size
        self.char_feature_size = builder.char_feature_size
        self.char_cell_size = builder.char_cell_size
        self.num_layers = builder.num_layers
        self.max_steps = builder.max_steps
        self.num_classes = builder.num_classes
        self.num_entity_classes = builder.num_entity_classes
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
        self.char_vocab_size = builder.char_vocab_size
        self.use_char_embeddings = builder.use_char_embeddings
        self.build_graph()

    def build_graph(self):
        with tf.Graph().as_default():
                with tf.name_scope('input_pipe_line'):
                    if self.oper_mode == RNNModel.OperMode.OPER_MODE_TEST:
                        self.xs = tf.placeholder(tf.float32,[1,None,self.feature_size],name='xs')
                        # pad's second argument can be seen as [[up, down], [left, right]]
                        if self.is_classifier:
                            self.zs = tf.placeholder(tf.int64, [None], name='zs')
                            self.ys = tf.placeholder(tf.int64, [1, None], name='ys')
                        else:
                            self.ys = tf.placeholder(tf.int64, [1, None], name='ys')

                        self.steps = tf.placeholder(tf.int64, [None], name='steps')
                    else:
                        decoder = TFDecoder.Builder(). \
                            set_feature_size(self.word_feature_size). \
                            set_num_epochs(self.epochs). \
                            set_path(self.read_path). \
                            set_shuffle_status(True). \
                            build()
                        self.steps, self.xs , self.ys, self.ws, self.wl, self.zs = tf.train.batch(tensors=decoder.dequeue(self.is_classifier),
                                                                                                  batch_size=self.batch_size,
                                                                                                  dynamic_pad=True,
                                                                                                  allow_smaller_final_batch=True,
                                                                                                  name='batch_processor')
                        self.global_step = tf.Variable(0, name="global_step", trainable=False)

                    xs = self.xs
                    ys = self.ys
                    zs = self.zs
                    ws = self.ws
                    wl = self.wl

                    self.wl_reshaped = tf.reshape(wl,[-1])

                    self.keepprob = tf.placeholder(tf.float32, [], name='keeprob')

                if self.use_char_embeddings is True:

                    with tf.name_scope('embedding_layer'):
                        embeddings = tf.get_variable(
                            name='word_embeddings',
                            dtype=tf.float32,
                            shape=[self.char_vocab_size, self.char_feature_size])

                    # char representation
                    # input weights
                    with tf.name_scope('char_input_layer'):
                        with tf.name_scope('Weigths'):
                            initalizer = tf.contrib.layers.xavier_initializer()
                            Wc = self.weight_variable([self.char_feature_size, self.char_cell_size], initializer=initalizer, name='Wc')
                            if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                                tf.summary.histogram('char_input_layer/Weights', Wc)
                        with tf.name_scope('Biases'):
                            Bc = self.bias_variable([self.char_cell_size], name='Bc')
                            if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                                tf.summary.histogram('char_input_layer/Biases', Bc)

                        dense = tf.sparse_tensor_to_dense(ws,name='dense')
                        self.dense = dense

                        # Batch x  Max Words x Max Char x Char Feature Size
                        char_vec = tf.nn.embedding_lookup(embeddings, dense,name='embedded_lookup')
                        self.char_vec = char_vec

                        char_vec = tf.reshape(char_vec,[-1,self.char_feature_size])

                        self.char_vec_reshaped = char_vec

                        char_rnn_inputs = tf.add(tf.matmul(char_vec, Wc), Bc)
                        # self.input_test = rnn_inputs

                        char_rnn_inputs = tf.nn.dropout(char_rnn_inputs, keep_prob=self.keepprob)

                        char_rnn_inputs = tf.reshape(char_rnn_inputs, [-1, tf.shape(dense)[-1], self.char_cell_size],
                                         name='char_rnn_inputs')

                        self.char_rnn_inputs = char_rnn_inputs

                        with tf.name_scope('char_rnn_layer'):
                            if self.cell_type == RNNModel.CellType.RNN_CEL_TYPE_LSTM:
                                char_cell_fw = tf.nn.rnn_cell.LSTMCell(self.char_cell_size, state_is_tuple=True, name='CharLSTMCell')
                                char_cell_bw = tf.nn.rnn_cell.LSTMCell(self.char_cell_size, state_is_tuple=True, name='CharLSTMCell')
                            else:
                                char_cell_fw = tf.nn.rnn_cell.GRUCell(self.char_cell_size, name='CharGRUCell')
                                char_cell_bw = tf.nn.rnn_cell.GRUCell(self.char_cell_size, name='CharGRUCell')

                            char_cell_fw = tf.nn.rnn_cell.DropoutWrapper(char_cell_fw, input_keep_prob=self.keepprob)
                            char_cell_bw = tf.nn.rnn_cell.DropoutWrapper(char_cell_bw, input_keep_prob=self.keepprob)

                            if self.cell_type == RNNModel.CellType.RNN_CEL_TYPE_LSTM:
                                char_cell_fw = tf.nn.rnn_cell.MultiRNNCell([char_cell_fw] * self.num_layers, state_is_tuple=True)
                                char_cell_bw = tf.nn.rnn_cell.MultiRNNCell([char_cell_bw] * self.num_layers, state_is_tuple=True)
                            else:
                                char_cell_fw = tf.nn.rnn_cell.MultiRNNCell([char_cell_fw] * self.num_layers)
                                char_cell_bw = tf.nn.rnn_cell.MultiRNNCell([char_cell_bw] * self.num_layers)

                            char_cell_fw = tf.nn.rnn_cell.DropoutWrapper(char_cell_fw, output_keep_prob=self.keepprob)
                            char_cell_bw = tf.nn.rnn_cell.DropoutWrapper(char_cell_bw, output_keep_prob=self.keepprob)

                            char_batch_size = tf.shape(char_rnn_inputs)[0]

                            self.char_batch_size = char_batch_size

                            # self.tf_batch_size = batch_size

                            char_state_fw = char_cell_fw.zero_state(char_batch_size, tf.float32)
                            char_state_bw = char_cell_bw.zero_state(char_batch_size, tf.float32)

                            if self.is_bidirectional is True:

                                char_outputs, char_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=char_cell_fw,
                                                                            cell_bw=char_cell_bw,
                                                                            inputs=char_rnn_inputs,
                                                                            sequence_length=tf.reshape(wl,[-1]),
                                                                            initial_state_fw=char_state_fw,
                                                                            initial_state_bw=char_state_bw,
                                                                            time_major=False)

                                self.char_rnn_outputs = tf.concat(char_outputs, 2)
                                self.char_final_state = char_final_state

                                char_feature_size =  2 * self.char_feature_size * self.num_layers




                                self.char_state_trans = tf.transpose(self.char_final_state,perm=[2,0,1,3])

                                self.char_rep = tf.reshape(self.char_state_trans,[tf.shape(self.char_rnn_outputs)[0], -1])
                                # (self.state_fw, self.state_bw) = self.final_state
                            else:

                                char_outputs, char_final_state = tf.nn.dynamic_rnn(char_cell_fw, char_rnn_inputs, sequence_length=tf.reshape(wl,[-1]),
                                                              initial_state=char_state_fw, time_major=False)

                                self.char_rnn_outputs = char_outputs
                                self.char_final_state = char_final_state

                                self.char_state_trans = tf.transpose(self.char_final_state, perm=[1, 0, 2])

                                self.char_rep = tf.reshape(self.char_state_trans, [tf.shape(self.char_rnn_outputs)[0], None])

                                char_feature_size = self.char_feature_size * self.num_layers

                                # self.state_fw = self.final_state


                    x = tf.concat((tf.reshape(xs,[-1,self.word_feature_size]), self.char_rep),axis=-1)

                    self.inputs = x

                    feature_size = char_feature_size + self.word_feature_size
                else:
                    x = tf.reshape(xs, [-1, self.word_feature_size])
                    self.inputs = x
                    feature_size = self.word_feature_size

                # word representation
                # input weights
                with tf.name_scope('input_layer'):
                    with tf.name_scope('Weigths'):
                        Win = self.weight_variable([feature_size,self.cell_size],name='W_in')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('input_layer/Weights',Win)
                    with tf.name_scope('Biases'):
                        Bin = self.bias_variable([self.cell_size], name='B_in')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('input_layer/Biases', Bin)



                    rnn_inputs = tf.add(tf.matmul(x,Win),Bin)

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

                with tf.name_scope('output_layer_1'):
                    with tf.name_scope('Weights'):
                        Wout_1 = self.weight_variable([cell_size,self.num_entity_classes],name='W_out_1')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('output_layer_1/Weights_1',Wout_1)
                    with tf.name_scope('Biases'):
                        Bout_1 = self.bias_variable([self.num_entity_classes],name='B_out_1')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('outpu_layer_1/Bias_1',Bout_1)

                with tf.name_scope('output_layer_2'):
                    with tf.name_scope('Weights'):
                        Wout_2 = self.weight_variable([cell_size,self.num_classes],name='W_out_2')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('output_layer_2/Weights_2',Wout_2)
                    with tf.name_scope('Biases'):
                        Bout_2 = self.bias_variable([self.num_classes],name='B_out_2')
                        if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                            tf.summary.histogram('outpu_layer_2/Bias_2',Bout_2)

                ntime_steps = tf.shape(rnn_outputs)[1]
                if self.is_classifier is True:
                        # get last rnn output
                        # self.test_rnn_outputs = tf.transpose(rnn_outputs,perm=[1,0,2])[-1]

                        # if self.is_timemajor is True:
                        #     rnn_outputs = rnn_outputs[-1]
                        # else:
                        self.idx0 = tf.range(tf.cast(batch_size,tf.int64))
                        self.idx1 = self.idx0 * tf.cast(tf.shape(tf.cast(rnn_outputs,tf.int64))[1],tf.int64)
                        self.idx2 =  self.idx1 + (self.steps - 1)


                        rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size])
                        self.last_rnn_output = tf.gather(rnn_outputs, self.idx2)


                else:
                        rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size])


                #for seq2seq calcuations
                self.rnn_outputs = rnn_outputs

                class_logits = tf.add(tf.matmul(self.last_rnn_output,Wout_2),Bout_2)

                entity_logits = tf.add(tf.matmul(rnn_outputs,Wout_1),Bout_1)

                # for seq2seq calculations
                self.class_logits = tf.nn.softmax(class_logits)
                self.entity_logits = tf.reshape(entity_logits,[-1, ntime_steps, self.num_entity_classes])

                # self.entity_logits = tf.nn.softmax(entity_logits)

                self.class_predictions = tf.argmax(self.class_logits,axis=-1)
                self.class_flat_labels = tf.reshape(zs, [-1])

                # self.entity_predictions = tf.argmax(self.entity_logits,axis=-1)

                self.entity_flat_labels = tf.reshape(ys, [-1])

                with tf.name_scope('cross_entropy'):

                    self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.class_logits, labels=self.class_flat_labels))

                    self.log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.entity_logits, ys,
                                                                                               tf.cast(self.steps,dtype=tf.int32))

                    self.entity_loss = tf.reduce_mean(-self.log_likelihood)

                    self.loss = self.entity_loss + self.class_loss
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

                    tf.summary.scalar('Class Cross Entropy', self.class_loss)
                    tf.summary.scalar('Entity Cross Entropy', self.entity_loss)

                with tf.name_scope('accuracy'):

                    entity_predictions , self.entity_scores = tf.contrib.crf.crf_decode(self.entity_logits, self.trans_params, tf.cast(self.steps,dtype=tf.int32))
                    self.entity_predictions = tf.cast(tf.reshape(entity_predictions,[-1]),dtype=tf.int64)
                    # self.entity_accuracy = tf.reduce_mean(tf.cast(tf.equal(
                    #     self.entity_flat_labels,self.entity_predictions, tf.float32)),name='entity_accuracy')

                    self.entity_accuracy = tf.reduce_mean(
                            tf.cast(tf.equal(self.entity_flat_labels, self.entity_predictions), tf.float32),
                            name='entity_accuracy')

                    self.class_accuracy = tf.reduce_mean(
                            tf.cast(tf.equal(self.class_flat_labels, self.class_predictions), tf.float32),
                            name='class_accuracy')

                    tf.summary.scalar('Entity Accuracy', self.entity_accuracy)
                    tf.summary.scalar('Class Accuracy', self.class_accuracy)
                    # weights = tf.reshape
                    # tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits=logits, labels=self.logits)
                if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                    with tf.name_scope('train'):
                        self.cls_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.class_loss,global_step=self.global_step,name='class_train_step')
                        self.entity_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.entity_loss,
                                                                                        global_step=self.global_step,
                                                                                          name='entity_train_step')
                        # self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                        #                                                                           global_step=self.global_step,
                        #                                                                           name='train_step')

                self.saver = tf.train.Saver(tf.global_variables(),keep_checkpoint_every_n_hours=1,max_to_keep=2)

                self.g_init = tf.global_variables_initializer()
                self.l_init = tf.local_variables_initializer()



                self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

                self.summary = tf.summary.merge_all()

                if self.oper_mode == RNNModel.OperMode.OPER_MODE_TRAIN:
                    self.summary_writer = tf.summary.FileWriter(self.logs_path + '/train' ,self.sess.graph)
                elif self.oper_mode == RNNModel.OperMode.OPER_MODE_EVAL:
                    self.summary_writer = tf.summary.FileWriter(self.logs_path + '/eval', self.sess.graph)


    def weight_variable(self, shape, initializer = None, name='weights'):
        if initializer is None:
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
            total_ent_acc = 0.0

            while not self.coord.should_stop():


                # dense, char_vec, char_vec_reshaped, char_rnn_inputs, char_rnn_outputs, char_final_state, \
                # char_state_trans, char_rep, xs, x, wl,wl_reshaped  = self.sess.run([self.dense,
                #                                                                                       self.char_vec,
                #                                                                                       self.char_vec_reshaped,
                #                                                                                       self.char_rnn_inputs,
                #                                                                                       self.char_rnn_outputs,
                #                                                                                       self.char_final_state,
                #                                                                                       self.char_state_trans,
                #                                                                                       self.char_rep,
                #                                                                                       self.xs,
                #                                                                                       self.inputs,
                #                                                                                       self.wl,
                #                                                                                       self.wl_reshaped],feed_dict)
                # print(np.shape(dense), np.shape(char_vec), np.shape(char_vec_reshaped), np.shape(char_rnn_inputs),
                #       np.shape(char_rnn_outputs), np.shape(char_final_state),
                #       np.shape(char_state_trans), np.shape(char_rep), np.shape(xs), np.shape(x),
                #       np.shape(wl), np.shape(wl_reshaped))

                xs, x  = self.sess.run([self.xs,self.inputs],feed_dict)
                print(np.shape(xs), np.shape(x))
                continue

                _,_,loss,  class_accuaracy, entity_accuracy, summary, final_state , entity_predictions, scores, ys   = self.sess.run([self.cls_train_step, self.entity_train_step, self.loss, self.class_accuracy, self.entity_accuracy, self.summary, self.final_state, self.entity_predictions, self.entity_scores, self.ys],feed_dict)
                total_loss += loss
                total_acc += class_accuaracy
                total_ent_acc += entity_accuracy
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
                # , entity_predictions, np.reshape(ys,[-1])
                print('Train: ',current_step,loss,class_accuaracy, entity_accuracy)
                if current_step % self.validation_step == 0:
                    self.summary_writer.add_summary(summary, current_step)
                    self.summary_writer.flush()
                    # print(seq_weights)
                    # print('Saving model params for step: ', current_step)

                    print('{} Loss: {} Class Accuarcy {} Entity Accuracy {}'.format(self.oper_mode, total_loss / count, total_acc / count, total_ent_acc/count))
                    total_loss = 0.0
                    total_acc = 0.0
                    total_ent_acc = 0.0
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
        total_ent_acc = 0.0

        summary = None
        try:
            feed_dict = {self.keepprob: keepprob}

            while not self.coord.should_stop():
                loss, class_accuracy,entity_accuracy, summary = self.sess.run([self.loss,self.class_accuracy,self.entity_accuracy, self.summary],feed_dict)
                total_loss += loss
                total_acc += class_accuracy
                total_ent_acc += entity_accuracy
                count += 1




        except Exception as err:
            print('{} Out of Range'.format(self.oper_mode))
        finally:
            self.coord.request_stop()
            if count > 0:
                self.summary_writer.add_summary(summary, curr_step)
                self.summary_writer.flush()
                print('{} Loss : {}  Class Accuracy: {} Entity Accuracy: {}'.format(self.oper_mode, total_loss/count, total_acc/count, total_ent_acc/count))
        self.coord.join(self.threads)
        self.threads = None

    def test(self,xs,steps):
        feed_dict = {self.keepprob: 1.0, self.xs : xs, self.steps : steps}

        result = self.sess.run([self.class_predictions, self.entity_predictions], feed_dict)

        print(result)

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
            self.word_feature_size = 300
            self.num_classes = 8
            self.num_entity_classes = 42
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
            self.char_feature_size = 50
            self.char_cell_size = 200
            self.char_vocab_size = 100
            self.use_char_embeddings = True

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

        def set_word_feature_size(self,val):
            self.word_feature_size = val
            return self


        def set_char_feature_size(self,val):
            self.char_feature_size = val
            return self


        def set_class_size(self,val):
            self.num_classes = val
            return self


        def set_entity_class_size(self,val):
            self.num_entity_classes = val
            return self

        def set_cell_size(self,val):
            self.cell_size = val
            return self

        def set_char_cell_size(self,val):
            self.char_cell_size = val
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

        def set_char_vocab_size(self, val):
            self.char_vocab_size = val
            return self

        def set_char_emb_status(self,flag):
            self.use_char_embeddings = flag
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




