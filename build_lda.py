# IMPORTS #

import logging
import warnings

import spacy
import gensim
import nltk
import itertools

import pandas as pd
import dill as pickle

import params_lda as par
from functions_lda import sent_to_words_generator, make_bigrams

warnings.filterwarnings("ignore", category=DeprecationWarning)
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# %%
# READ AND PREPROCESS DATA #

stop_words = list(set().union(
                pd.read_csv(par.CUSTOM_STOPWORDS_PATH)['word'].tolist(),
                nltk.corpus.stopwords.words('english')
                ))

input_df = pd.read_csv(par.INPUT_PATH,
                       sep=';',
                       encoding='latin1',
                       usecols=[par.ID_COL, par.RAW_TEXT_COL])

data_sentences = input_df[par.RAW_TEXT_COL].tolist()
data_words = list(sent_to_words_generator(data_sentences))

data_lemmas = [
        [token.lemma_ for token in nlp(" ".join(text))
         if token.pos_ in par.ALLOWED_POS
         and token.lemma_ not in stop_words
         and token.text not in stop_words]
        for text in data_words]

data_processed = make_bigrams(data_lemmas)
data_proc_words = itertools.chain.from_iterable(data_processed)
freq_dist = nltk.FreqDist(data_proc_words)
most_common_words = list(list(zip(*freq_dist.most_common(10)))[0])

skip_sent = [0 if len(sent) >= par.MIN_SENT_LENGTH else 1
             for sent in data_processed]
data_clean = [sent for sent in data_processed
              if len(sent) >= par.MIN_SENT_LENGTH]

save_df = pd.DataFrame({par.ID_COL: input_df[par.ID_COL],
                        par.RAW_TEXT_COL: input_df[par.RAW_TEXT_COL],
                        'text_proc': data_processed,
                        'skip': skip_sent})

# %%
# BUILD & PICKLE MODELS #

id2word = gensim.corpora.Dictionary(data_clean)
id2word.filter_extremes(
        no_below=par.EXT_LOW,
        no_above=par.EXT_HI)

corpus = [id2word.doc2bow(text) for text in data_clean]
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=par.NUM_TOPICS,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=10,
                                            passes=par.NUM_ITERATIONS,
                                            alpha='auto',
                                            eta='auto',
                                            per_word_topics=True)
coherence_model = gensim.models.CoherenceModel(model=lda_model,
                                               texts=data_clean,
                                               dictionary=id2word,
                                               coherence='c_v',
                                               processes=1)

print('\nModel Coherence Score: ', coherence_model.get_coherence())

data_to_pickle = {'df': save_df,
                  'dict': id2word,
                  'corpus': corpus,
                  'model': lda_model,
                  'coherence': coherence_model,
                  'freq': freq_dist}

with open(par.LDA_PICKLE_PATH, 'wb') as f:
    pickle.dump(data_to_pickle, f)
