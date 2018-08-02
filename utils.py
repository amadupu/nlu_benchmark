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


def generate_data(path,is_test = False):

    if is_test is True:
        pattern = re.compile(r'validate.*[.]json')
    else:
        pattern = re.compile(r'train.*full[.]json')

    categories = dict()
    category_index = 0


    output = list()


    for r,d,f in os.walk(path):
        for filename in f:



            #define data category train/test
            if not pattern.match(filename):
                continue

            label = 0  # none


            # define lable category Book Reservation/Add Playlist/etc

            label_name = os.path.basename(r)

            if label_name not in categories:
                category_index += 1
                categories[label_name] = category_index

            label = categories[label_name]

            filepath = os.path.join(r,filename)


            data = None

            try:
                print('Loading {} {} ..'.format(label_name, filename),end='')
                with open(filepath,encoding='utf-8', mode='r') as fp:
                    data = json.load(fp)
                print('SUCCESS')
            except Exception as err:
                print('FAILED! ',err)

            if data is None:
                continue


            # form the data tuples

            for topic, values in data.items():

                for value in values:
                    data_list = value['data']
                    sent = r''
                    for elem in data_list:
                        sent += elem['text']

                    output.append(tuple((sent,label)))

    shuffle(output)
    return (categories, output)







