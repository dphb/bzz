# IMPORTS #

import logging
import warnings

import gensim
import itertools
import pandas as pd

import params_lda as par

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

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


def generate_outputs(lda_model, lda_df, lda_dict, save_output=True):
    # Create document-topic distribution df
    topic_distr = []

    for i, row in lda_df.iterrows():
        if row['skip'] == 1:
            to_append = [row[par.ID_COL], -1, -1]
            topic_distr.append(to_append)
        else:
            text_bow = lda_dict.doc2bow(row['text_proc'])
            for top, pct in lda_model.get_document_topics(text_bow):
                topic_distr.append([row[par.ID_COL], top, pct])

    topic_distr_df = pd.DataFrame(topic_distr,
                                  columns=[par.ID_COL, 'topic_id', 'topic_pct'])

    # Create best topic representatives df
    best_ids = {}
    topic_distr_filt = topic_distr_df.loc[topic_distr_df['topic_id'] != -1]

    for topic_id, grp in topic_distr_filt.groupby('topic_id'):
        grp_sorted = grp.sort_values('topic_pct', ascending=False)
        best_ids[topic_id] = grp_sorted[:par.NUM_TOPEXAMPLES][
                par.ID_COL].tolist()

    best_texts = []
    for topic, ids in best_ids.items():
        raw_texts = [topic] + [lda_df[lda_df[
                par.ID_COL] == i].iloc[0][par.RAW_TEXT_COL] for i in ids]
        best_texts.append(raw_texts)

    name_strings = [str(i+1) + ' best' for i in range(par.NUM_TOPEXAMPLES)]
    best_texts_df = pd.DataFrame(best_texts,
                                 columns=['topic_id'] + name_strings)

    # Create topic words & probs df
    topic_words = []
    for t in range(par.NUM_TOPICS):
        words = [word for word, prob in lda_model.show_topic(t)]
        probs = [prob for word, prob in lda_model.show_topic(t)]
        topic_words.append([t, words, probs])

    topic_words_df = pd.DataFrame(topic_words,
                                  columns=['topic_id', 'top_words', 'word_pct'])

    # Return outputs
    topic_df = topic_words_df.merge(best_texts_df,
                                   on='topic_id')
    if(save_output):
        topic_df.to_csv(par.TOPICS_SAVE_PATH, index=False)
        topic_distr_df.to_csv(par.DOC_DISTR_SAVE_PATH, index=False)

    return topic_df, topic_distr_df
