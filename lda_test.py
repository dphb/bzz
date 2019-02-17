# IMPORTS #

import logging
import warnings
import os
import spacy
import gensim
import math

import pandas as pd
import dill as pickle
import nltk.corpus

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)

# %%
# PARAMETERS #

input_path = 'full_text.csv'
custom_stopwords_path = 'stopwords.csv'
mallet_path = 'C:/mallet-2.0.8/bin/mallet'
lda_obj_path = 'LDA_objects.pickle'

id_col = 'global_id'
text_col = 'text_raw'
allowed_pos = ['NOUN', 'ADJ', 'VERB', 'ADV']

min_sent_length = 2
num_topics = 15
num_passes = 20
ext_low = 0.05
ext_hi = 0.5

# %%
# FUNCTIONS #


def sent_to_words_generator(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def make_bigrams(texts, min_count=5, threshold=50):
    bigram = gensim.models.phrases.Phrases(texts,
                                           min_count,
                                           threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return([bigram_mod[doc] for doc in texts])


def make_trigrams(texts,
                  min_count_bi=5, threshold_bi=50,
                  min_count_tri=5, threshold_tri=20):
    texts_bi = make_bigrams(texts, min_count_bi, threshold_bi)
    texts_tri = make_bigrams(texts_bi, min_count_tri, threshold_tri)
    return(texts_tri)


# %%
# READ AND PREPROCESS DATA #

stop_words = list(set().union(
                pd.read_csv(custom_stopwords_path)['word'].tolist(),
                nltk.corpus.stopwords.words('english')
                ))
input_df = pd.read_csv(input_path,
                       sep=';',
                       encoding='latin1',
                       usecols=[id_col, text_col])

data_sentences = input_df[text_col].tolist()
data_words = list(sent_to_words_generator(data_sentences))
data_processed = [
        [token.lemma_ for token in nlp(" ".join(text))
         if token.pos_ in allowed_pos
         and token.lemma_ not in stop_words
         and token.text not in stop_words]
        for text in data_words]
data_clean = [sent if len(sent) >= min_sent_length else []
              for sent in make_bigrams(data_processed)]

# %%
# BUILD & PICKLE MODELS #

id2word = gensim.corpora.Dictionary(data_clean)
id2word.filter_extremes(
        no_below=math.floor(ext_low*len(data_clean)),
        no_above=ext_hi)
corpus = [id2word.doc2bow(text) for text in data_clean]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=num_passes,
                                            alpha='auto',
                                            eta='auto',
                                            per_word_topics=True)
coherence_model = gensim.models.CoherenceModel(model=lda_model,
                                               texts=data_clean,
                                               dictionary=id2word,
                                               coherence='c_v',
                                               processes=1)

print('\nModel Coherence Score: ', coherence_model.get_coherence())

data_to_pickle = {'texts': data_clean,
                  'ids': input_df[id_col].tolist(),
                  'dict': id2word,
                  'corpus': corpus,
                  'model': lda_model,
                  'coherence': coherence_model}

with open(lda_obj_path, 'wb') as f:
    pickle.dump(data_to_pickle, f)
