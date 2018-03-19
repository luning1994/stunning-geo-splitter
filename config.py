HIDDEN_DIM = 80
LEARNING_RATE = 0.001
NEPOCH = 500
VOCABULARY_SIZE_LIMIT = 8000
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENT_START_TOKEN = "SENTENCE_START"
SENT_END_TOKEN = "SENTENCE_END"
SPLIT_TOKEN = "SPLIT_TOKEN"
ban_list = ['vn', 'hcm vn', 'hn', 'hn vn', 'ho chi minh vn', 'ha noi vn',
                'hcm', 'remote location', 'ha noi']
DATA_DIR = "data\data.p"
MODEL_DIR = "data\\gru-epoch30.npz"
INPUT_DIR = "preprocessed.xlsx"
OUTPUT_DIR = "output\\result-gru-numpen.xlsx"
# MODEL_FILE = "data\\rnn-theano-80-3241-2016-09-01-17-08-54.npz" # with 30% total accuracy
