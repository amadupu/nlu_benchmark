
#input_data_source_path = r'D:\VM\vmshare\projects\nlu-benchmark\2017-06-custom-intent-engines\BookRestaurant'
input_data_source_path = r'/home/unnathi_madupu/projects/nlu-benchmark/2017-06-custom-intent-engines'
spacy_model_path=r'/home/unnathi_madupu/projects/corpus/glove'

# hyperparmeters
word_feature_size=300
char_feature_size = 50
pos_feature_size = 10




use_char_embeddings = True
char_vocab_size = 150
max_steps = 100  #deprecated
train_epochs = 100
cell_size = 512
char_cell_size = 50
batch_size = 100
num_classes = 8
num_entity_classes=42
num_layers = 3
learning_rate = 1e-4
model_name = 'spacy_default_ent'
state_feeback = False # may cause an exception when processing the last batch
bi_directional=True
validation_step = 100
is_classifer = True
time_major = False # deprecated
model_path = 'model'
logs_path = 'logs'
keep_prob = 0.85
