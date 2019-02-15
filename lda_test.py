# IMPORTS #

import logging
import warnings
import re
import os

import numpy as np
import pandas as pd

from pprint import pprint

import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
from nltk.corpus import stopwords

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# %%
# PARAMETERS #

input_path = 'full_text.csv'
id_col = 'global_id'
text_col = 'text_raw'
allowed_pos = ['NOUN', 'ADJ', 'VERB', 'ADV']
custom_stopwords_path = 'stopwords.csv'
os.environ['MALLET_HOME'] = "C:\\mallet-2.0.8\\"
mallet_path = "C:\\mallet-2.0.8\\bin\\mallet"
num_topics = 15

# %%
# FUNCTIONS #


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))


def make_bigrams(texts, min_count=5, threshold=100):
    bigram = gensim.models.phrases.Phrases(data_words,
                                           min_count,
                                           threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]


# %%
# READ AND PREPROCESS DATA #

stop_words = list(
        set().union(
                pd.read_csv(custom_stopwords_path)['word'].tolist(),
                stopwords.words('english')
                )
        )
df = pd.read_csv(input_path,
                 sep=';',
                 encoding='latin1',
                 usecols=[id_col, text_col])

data_sentences = df[text_col].tolist()
data_words = list(sent_to_words(data_sentences))
data_bigrammed = make_bigrams(data_words)
data_clean = [
        [token.lemma_ for token in nlp(" ".join(text))
         if token.pos_ in allowed_pos
         and token.lemma_ not in stop_words
         and token.text not in stop_words]
        for text in data_bigrammed
        ]

id2word = corpora.Dictionary(data_clean)
corpus = [id2word.doc2bow(text) for text in data_clean]
ldamallet = LdaMallet(
        mallet_path,
        corpus=corpus,
        num_topics=num_topics,
        id2word=id2word)
