from rnn_model import RNNModel
import threading
import os

# hyper parameters

feature_size=300
max_steps = 100
train_epochs = 100
cell_type = RNNModel.CellType.RNN_CELL_TYPE_GRU
cell_size = 512
batch_size = 50
num_classes = 8
num_layers = 3
learning_rate = 1e-4
model_name = 'spacy_default'

state_feeback = False # may cause an exception when processing the last batch

bi_directional=True
validation_step = 30
is_classifer = True

time_major = False # deprecated

if is_classifer is True:
    model_path = 'model'
    logs_path = 'logs'
else:
    model_path = 'model'
    logs_path = 'logs'

keep_prob = 0.85



if __name__ == '__main__':
    def child_process(curr_step):
        print('Starting child process')
        model = RNNModel.Builder().set_max_steps(max_steps). \
            set_feature_size(feature_size). \
            set_read_path(os.path.join('records','eval')). \
            set_epochs(1). \
            set_cell_type(cell_type). \
            set_cell_size(cell_size). \
            set_batch_size(batch_size). \
            set_class_size(num_classes). \
            set_layer_size(num_layers). \
            set_model_path(model_path). \
            set_model_name(model_name).\
            set_logs_path(logs_path). \
            set_bi_directional(bi_directional). \
            set_classifer_status(is_classifer). \
            set_state_feedback(state_feeback). \
            set_time_major(time_major).\
            set_oper_mode(RNNModel.OperMode.OPER_MODE_EVAL). \
            build()
        model.evaluate(curr_step=curr_step)

    def evaluator(current_count):
        print('Evaluating for step: ',current_count)
        # eval_model.evaluate()
        thread = threading.Thread(target=child_process,args=(current_count,),daemon=False)
        thread.start()
        thread.join()


        # eval_model.evaluate()

    train_model = RNNModel.Builder().set_max_steps(max_steps).\
        set_feature_size(feature_size).\
        set_read_path(os.path.join('records', 'train')). \
        set_epochs(train_epochs).\
        set_cell_type(cell_type).\
        set_cell_size(cell_size).\
        set_batch_size(batch_size).\
        set_class_size(num_classes).\
        set_layer_size(num_layers).\
        set_learning_rate(learning_rate). \
        set_model_path(model_path). \
        set_model_name(model_name). \
        set_logs_path(logs_path).\
        set_eval_fn(evaluator). \
        set_time_major(time_major). \
        set_state_feedback(state_feeback). \
        set_bi_directional(bi_directional) .\
        set_classifer_status(is_classifer).\
        set_oper_mode(RNNModel.OperMode.OPER_MODE_TRAIN). \
        set_validation_step(validation_step).\
        build()

    train_model.train(keep_prob)

#         set_eval_fn(evaluator). \







