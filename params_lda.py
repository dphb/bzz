# INPUT PARAMS #

INPUT_PATH = 'full_text.csv'
CUSTOM_STOPWORDS_PATH = 'stopwords.csv'
MALLET_PATH = 'C:/mallet-2.0.8/bin/mallet'

ID_COL = 'global_id'
RAW_TEXT_COL = 'text_raw'

# MODEL PARAMS #

ALLOWED_POS = ['NOUN', 'ADJ', 'VERB', 'ADV']

MIN_SENT_LENGTH = 5
NUM_TOPICS = 5
NUM_ITERATIONS = 20
EXT_LOW = 5
EXT_HI = 0.75

# OUTPUT PARAMS #

NUM_TOPEXAMPLES = 3

LDA_PICKLE_PATH = 'LDA_objects.pickle'
TOPICS_SAVE_PATH = 'topics.csv'
DOC_DISTR_SAVE_PATH = 'doc_distr.csv'
