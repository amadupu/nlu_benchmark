
input_data_source_path = r'/home/arun_madupu/projects/nlu-benchmark/2017-06-custom-intent-engines'
spacy_model_path=r'/home/arun_madupu/projects/corpus/glove'

# hyperparmeters
feature_size=600
max_steps = 100
train_epochs = 100
cell_size = 512
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
