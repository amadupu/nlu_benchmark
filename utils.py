import os
import spacy
import re
import json
from random import shuffle
import time
import datetime

def get_file_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S%f')


def clean_dir(path):
    for r,d,f in os.walk(path):
        for filename in f:
            filename = os.path.join(r,filename)
            if os.path.isfile(filename):
                os.remove(filename)


def preprocess_data(segment):

    # multiple spaces with single space
    pattern = re.compile(r'\s+')
    segment = pattern.sub(r' ', segment)

    pattern = re.compile(r'\bwe[\']d\b', re.IGNORECASE)
    segment = pattern.sub('we would', segment)

    pattern = re.compile(r'\b[iI][\']d\b')
    segment = pattern.sub('i would', segment)
    # remove skip gramps

    # pattern = re.compile(r'[\']\w+')
    # segment = pattern.sub('', segment)


    pattern = re.compile(r'[^\w\s]+')
    segment = pattern.sub('', segment)

    pattern = re.compile(r'\bisnt\b', re.IGNORECASE)
    segment = pattern.sub('is not', segment)

    pattern = re.compile(r'\bitll\b', re.IGNORECASE)
    segment = pattern.sub('it will', segment)

    pattern = re.compile(r'\btheres\b', re.IGNORECASE)
    segment = pattern.sub('there is', segment)

    pattern = re.compile(r'\bwheres\b', re.IGNORECASE)
    segment = pattern.sub('where is', segment)

    pattern = re.compile(r'\bdidnt\b', re.IGNORECASE)
    segment = pattern.sub('did not', segment)

    pattern = re.compile(r'\baint\b', re.IGNORECASE)
    segment = pattern.sub('Iam not', segment)

    pattern = re.compile(r'\bim\b', re.IGNORECASE)
    segment = pattern.sub('Iam', segment)

    pattern = re.compile(r'\bcant\b', re.IGNORECASE)
    segment = pattern.sub('can not', segment)

    pattern = re.compile(r'\bcant\b', re.IGNORECASE)
    segment = pattern.sub('can not', segment)

    pattern = re.compile(r'\bwhens\b', re.IGNORECASE)
    segment = pattern.sub('when is', segment)

    pattern = re.compile(r'\btheyre\b', re.IGNORECASE)
    segment = pattern.sub('they are', segment)

    pattern = re.compile(r'\bwhats\b', re.IGNORECASE)
    segment = pattern.sub('what is', segment)

    pattern = re.compile(r'\bhows\b', re.IGNORECASE)
    segment = pattern.sub('how is', segment)

    pattern = re.compile(r'\bgonna\b', re.IGNORECASE)
    segment = pattern.sub('going to', segment)

    pattern = re.compile(r'\byoure\b', re.IGNORECASE)
    segment = pattern.sub('you are', segment)

    pattern = re.compile(r'\bdont\b', re.IGNORECASE)
    segment = pattern.sub('do not', segment)

    pattern = re.compile(r'\bthats\b', re.IGNORECASE)
    segment = pattern.sub('that is', segment)

    pattern = re.compile(r'\bwed\b', re.IGNORECASE)
    segment = pattern.sub('wednesday', segment)

    pattern = re.compile(r'\bid\b', re.IGNORECASE)
    segment = pattern.sub('identity', segment)

    return segment


def generate_data(path,is_test = False):

    if is_test is True:
        filename_pattern = re.compile(r'validate.*[.]json')
    else:
        filename_pattern = re.compile(r'train.*full[.]json')


    headers = dict()
    entities = dict()
    entity_index = 2 # 0, 1, 2 are reserved
    categories = dict()
    category_index = 0


    output = list()


    for r,d,f in os.walk(path):
        for filename in f:
            # define data category train/test
            if not filename_pattern.match(filename):
                continue

            label_name = os.path.basename(r)

            if label_name not in categories:
                category_index += 1
                categories[label_name] = category_index

            label = categories[label_name]

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
                        entity_id = 0
                        if 'entity' in elem:
                            entity = elem['entity']
                            if entity not in entities:
                                entity_index += 1
                                entities[entity] = entity_index
                            entity_id = entities[entity]

                        # preprocessing

                        pattern = re.compile(r'\bwed\b', re.IGNORECASE)

                        if pattern.search(segment):
                            print('{} Wed: {}'.format(filename, segment))

                        pattern = re.compile(r'\b[iI]d\b')

                        if pattern.search(segment):
                            print('{} Id: {}'.format(filename, segment))

                        pattern = re.compile(r'\baint\b', re.IGNORECASE)

                        if pattern.search(segment):
                            print('{} Aint: {}'.format(filename, segment))

                        segment = preprocess_data(segment)

                        segment = segment.split()

                        seg_len = len(segment)

                        for index, word in enumerate(segment):

                            if entity_id != 0:

                                if seg_len == 1:
                                    sent.append((word, entity_id))  # Entity Type
                                elif index == 0:
                                    sent.append((word, 1))  # Begin
                                elif index < seg_len - 1 or seg_len == 2:
                                    sent.append((word, entity_id))  # Entity Type
                                else:
                                    sent.append((word, 2))  # End
                            else:
                                sent.append((word, 0))

                    output.append(tuple((sent,label)))

    #with open('output.txt','w',encoding='utf-8') as fp:
    #    for item in output:
    #       fp.write(str(item) + '\n')

    shuffle(output)

    headers['category_map'] = categories
    headers['entity_map'] = entities
    return (headers, output)







