from rnn_model import RNNModel
import spacy, json
import numpy as np
from utils import preprocess_data
from paths import *


if __name__ == '__main__':

    model = RNNModel.Builder().set_max_steps(max_steps). \
        set_word_feature_size(word_feature_size + pos_feature_size). \
        set_cell_type(RNNModel.CellType.RNN_CELL_TYPE_GRU). \
        set_cell_size(cell_size). \
        set_batch_size(1). \
        set_class_size(num_classes). \
        set_entity_class_size(num_entity_classes). \
        set_layer_size(num_layers). \
        set_model_path(model_path). \
        set_model_name(model_name). \
        set_time_major(time_major).\
        set_bi_directional(bi_directional). \
        set_state_feedback(state_feeback). \
        set_classifer_status(is_classifer). \
        set_char_feature_size(char_feature_size). \
        set_char_cell_size(char_cell_size). \
        set_char_vocab_size(char_vocab_size). \
        set_oper_mode(RNNModel.OperMode.OPER_MODE_TEST). \
        build()

    model.init_graph()

    nlp = spacy.load(spacy_model_path)
    nlp_pos = spacy.load('en_core_web_sm')


    headers = None
    with open('headers.json','r') as fp:
        headers = json.load(fp)

    if headers is None:
        raise Exception('Unable to read header information')

    categ_map = dict()
    ent_map = dict()


    categories = headers['category_map']


    for key, value in categories.items():
        categ_map[value] = key


    entities = headers['entity_map']
    for key, value in entities.items():
        ent_map[value] = key

    while True:
        print('INPUT>', end='')
        data= input()

        try:

            with open('vocab_char.txt') as fp:
                vocab = json.load(fp)

            char_vocab = vocab['vocab']



            if data.lower() == 'exit' or data.lower() == 'quit':
                break

            word_list = data.split()
            word_vec = list()
            word_len = list()
            for word in word_list:

                char_vec = list()
                for ch in word:
                    char_vec.append(char_vocab.index(ch))

                word_vec.append(char_vec)
                word_len.append(len(char_vec))

            ws = np.zeros([len(word_vec), len(max(word_vec, key=lambda x: len(x)))],dtype=np.int64)

            for i, j in enumerate(word_vec):
                ws[i][0:len(j)] = j



            data = data.lower()   

            data = preprocess_data(data)

            word_list = data.split()

            doc = nlp_pos(data)

            if len(word_list) != len(doc):
                raise Exception('Doc vs Input Length Mismatch  ', len(word_list), len(doc))

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

                if not nlp(pos)[0].has_vector:
                    raise Exception('Invalid vector for pos: ', pos )

                pos_vector = nlp(pos)[0].vector[:pos_feature_size]
                word_vector = nlp(token.text)[0]

                if not word_vector.has_vector:
                   token_vector = nlp('UNK')[0].vector
                else:
                   token_vector = word_vector.vector 

                token_vector = token_vector[:word_feature_size]

                # xs.append(token.vector)
                if pos_feature_size > 0:
                    xs.append(np.concatenate((token_vector,pos_vector),axis=-1))
                else:
                    xs.append(token_vector)

                # if not token.is_punct and not token.is_stop and not token.is_space and token.has_vector:
                #     xs.append(token.vector)

            steps = len(xs)
            xs = np.reshape(xs, [1, steps, len(xs[0])])
            ws = np.reshape(ws, [1 ,steps, len(ws[0])])
            wl = np.reshape(word_len, [1, steps])


            result = model.test(xs, ws, wl, [steps])


            output = dict()

            entities_map = dict()

            prev = 0
            ent_list = list()
            capture = list()
            for i, val in enumerate(result[1]):

                if val == 0:
                    if prev == 0:
                        capture.clear()
                    elif prev == 1:
                        capture.clear()
                    elif prev == 2:
                        if len(capture) != 0:
                            raise Exception('Invalid 0 check with prev 2 for capture: ',capture)
                    elif len(capture) > 0:
                        ent_list.append((ent_map[prev], ''.join([word_list[x] + ' ' for x in capture])))
                        capture.clear()
                    elif len(capture) == 0:
                        pass
                    else:
                        raise Exception('Invalid Capture for case 0')

                elif val == 1:
                    if prev == 0:
                        capture.clear()
                        capture.append(i)
                    elif prev == 1:
                        capture.clear()
                        capture.append(i)
                    elif prev == 2:
                        if len(capture) != 0:
                            raise Exception('Invalid 1 check with prev 2 for capture: ',capture)
                        capture.append(i)

                    elif len(capture) > 0:
                        ent_list.append((ent_map[prev],''.join([word_list[x] + ' ' for x in capture])))
                        capture.clear()
                        capture.append(i)

                    else:
                        raise Exception('Invalid Capture for case 1')

                elif val == 2:
                    if prev == 0:
                        capture.clear()

                    elif prev == 1:
                        capture.clear()

                    elif prev == 2:
                        if len(capture) != 0:
                            raise Exception('Invalid 2 check with prev 2 for capture: ',capture)

                    elif len(capture) >= 2:
                        capture.append(i)
                        ent_list.append((ent_map[prev], ''.join([word_list[x] + ' ' for x in capture])))
                        capture.clear()

                    else:
                        raise Exception('Invalid Capture for case 2')


                else:
                    if prev == 0:
                        capture.append(i)
                        ent_list.append((ent_map[val], ''.join([word_list[x] + ' ' for x in capture])))
                        capture.clear()

                    elif prev == 1:
                        capture.append(i)

                    elif prev == 2:
                        if len(capture) != 0:
                            raise Exception('Invalid 2 check with prev 2 for capture: ',capture)
                        capture.append(i)
                        ent_list.append((ent_map[val], ''.join([word_list[x] + ' ' for x in capture])))
                        capture.clear()

                    elif val != prev:
                        if capture:
                            ent_list.append((ent_map[prev], ''.join([word_list[x] + ' ' for x in capture])))
                        capture.clear()
                        capture.append(i)
                    else:
                        capture.append(i)

                prev = val

                if i == len(word_list) - 1:
                    if prev > 2:
                        if capture:
                            ent_list.append((ent_map[prev], ''.join([word_list[x] + ' ' for x in capture])))
                            capture.clear()




            for ent in ent_list:
                key, val = ent
                val = val.strip()
                if key in entities_map:
                    if type(entities_map[key]) is list:
                        entities_map[key].append(val)
                    else:
                        entities_map[key] = [entities_map[key]]
                        entities_map[key].append(val)
                else:
                    entities_map[key] = val




            output['intent'] = categ_map[int(result[0])].split()[0]
            output['entities'] = entities_map

            output = json.dumps(output)

            print('OUTPUT> {}'.format(output))

        except Exception as err:

            print('OUTPUT> {}'.format(err))
