from rnn_model import RNNModel
import spacy, json
import numpy as np


feature_size=300
max_steps = 100
cell_type = RNNModel.CellType.RNN_CELL_TYPE_GRU
cell_size = 512
batch_size = 1
num_classes = 8
num_layers = 3
model_name = 'spacy_default'
is_classifer = True
model_path = 'model'
time_major = False
bi_directional=True
keep_prob = 1.0
state_feeback = False


if __name__ == '__main__':

    model = RNNModel.Builder().set_max_steps(max_steps). \
        set_feature_size(feature_size). \
        set_cell_type(RNNModel.CellType.RNN_CELL_TYPE_GRU). \
        set_cell_size(cell_size). \
        set_batch_size(batch_size). \
        set_class_size(num_classes). \
        set_layer_size(num_layers). \
        set_model_path(model_path). \
        set_model_name(model_name). \
        set_time_major(time_major).\
        set_bi_directional(bi_directional). \
        set_state_feedback(state_feeback). \
        set_classifer_status(is_classifer). \
        set_oper_mode(RNNModel.OperMode.OPER_MODE_TEST). \
        build()

    model.init_graph()

    nlp = spacy.load('en_core_web_lg')


    categories = None
    with open('headers.json','r') as fp:
        categories = json.load(fp)

    if categories is None:
        raise Exception('Unable to read header information')

    header_map = dict()

    for key, value in categories.items():
        header_map[value] = key


    while True:
        print('INPUT>', end='')
        data= input()

        if data.lower() == 'exit' or data.lower() == 'quit':
            break
            
        doc = nlp(data)

        xs = list()
        for token in doc:
            if not token.is_punct and not token.is_stop and not token.is_space and token.has_vector:
                xs.append(token.vector)

        steps = len(xs)
        xs = np.reshape(xs, [1, steps, feature_size])

        result = model.test(xs, [steps])

        print('OUTPUT> {}'.format(header_map[int(result)].split()[0]))





