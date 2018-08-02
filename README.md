# nlu_benchmark
NLU Benchmarking Exercise

# Prerequisites
    Python 3.5+
    Tensorflow 1.9.0+ (CPU/GPU)



# Downloads NLU repository
    git clone https://github.com/snipsco/nlu-benchmark.git
NOTE: this contains the data for trainngi and validation



# Edit paths.py for input data path
    input_data_source_path = r'<nlu_path>/2017-06-custom-intent-engines'


# Prepare the Setup
    run 'python3 setup.py'
#NOTE: This step will initialize some internal directory structure for storing logs, model information and TFRecords




# Generate TFRecords for the input data
# This step formalizes the input data from json file to TFRecord format used in tensorflow
# User is advised to look into the sample implementation of data_encoder.py for more details

    run 'python3 data_encoder.py'

NOTE-1: the record files will be placed at 'records/eval' and 'records/train' directory
NOTE-2: This step will take approximately 2 min for it to complete the generation of TF records from all the train set.
NOTE-3: Please observe for the following Logs and their status


# Training procedure

    run 'python3 rnn_test.py'
NOTE: Please refer to runn_test.py for the various hyper parametres applicable
NOTE: Batch Validation is performed periodically for every 'validation_steps'


# Evaluation Procedure

run 'python3 rnn_eval.py'

NOTE: After sufficient Training and Cross Validation Accuracy, you can manually stop the training and perform evaluation
NOTE: A INPUT prompt is provided where user can specify the input, and result will be displayed in OUTPUT prompt



# Reset (Optional)

incase you want to drop every transient data (logs, model, records ) and start a fresh
run 'python3 reset.py'


# Other Information
visit https://www.slideshare.net/KonstantinSavenkov/nlu-intent-detection-benchmark-by-intento-august-2017




