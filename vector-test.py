import spacy, re, os, json
from enum import Enum

model_path = r'/home/arun_madupu/projects/corpus/glove'

input_path = r'/home/arun_madupu/projects/nlu-benchmark/2017-06-custom-intent-engines'




def validate(path,is_test=False):

    nlp = spacy.load(model_path)

    if is_test is True:
        filename_pattern = re.compile(r'validate.*[.]json')
    else:
        filename_pattern = re.compile(r'train.*full[.]json')


    valid_words  = set()

    invalid_words = set()

    for r,d,f in os.walk(input_path):
        for filename in f:
            # define data category train/test
            if not filename_pattern.match(filename):
                continue

            label_name = os.path.basename(r)

            filepath = os.path.join(r, filename)

            data = None

            try:
                print('Loading {} {} ..'.format(label_name, filename), end='')
                with open(filepath, encoding='utf-8', mode='r') as fp:
                    data = json.load(fp)
                print('SUCCESS')
            except Exception as err:
                print('FAILED! ', err)

            if data is None:
                continue

            for topic, values in data.items():
                for value in values:
                    data_list = value['data']
                    sent = list()
                    for elem in data_list:
                        segment = elem['text']
                        segment = segment.split()
                        for _, word in enumerate(segment):
                            if nlp(word).has_vector is False:
                                invalid_words.add(word)
                            else:
                                valid_words.add(word)


    print('Summary: Valid Words: {} Invalid Words: {}'.format(len(valid_words), len(invalid_words)))


    with open('invalid-words.txt','w',encoding='utf-8') as fp:
        for x in invalid_words:
            fp.write(x)






