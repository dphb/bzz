# IMPORTS #

import logging
import warnings
import os

import pandas as pd
import dill as pickle
import itertools

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.ERROR)

# %%
# PARAMETERS #

lda_obj_path = 'LDA_objects.pickle'
top_save_path = 'top_words.csv'
top_distr_save_path = 'top_distr.csv'

# %%
# FUNCTIONS #


def topics_to_df(lda_model, topn=10):
    words_probs_list = [[t,
                         [x[0] for x in lda_model.show_topic(t, topn=10)],
                         [x[1] for x in lda_model.show_topic(t, topn=10)]]
                        for t in range(lda_model.num_topics)]
    return(pd.DataFrame(words_probs_list,
                        columns=['topic', 'top_words', 'prob']))


def doc_topic_distr_to_df(lda_model, id2word, data_clean, doc_ids):
    top_distr_list = [
            [list((doc_ids[i],) + tup) if len(data_clean[i]) > 0
             else list((doc_ids[i],) + (-1, -1))
             for tup in lda_model.get_document_topics(
                     id2word.doc2bow(data_clean[i])
                     )]
            for i in range(len(data_clean))]
    top_distr_list_long = itertools.chain.from_iterable(top_distr_list)
    return(pd.DataFrame(top_distr_list_long,
                        columns=['global_id', 'topic_id', 'topic_pct']
                        ).drop_duplicates())


# %%
# LOAD MODEL SPECS #

with open(lda_obj_path, 'rb') as f:
    lda_data = pickle.load(f)

out1 = topics_to_df(lda_data['model'])
out1.to_csv(top_save_path)

out2 = doc_topic_distr_to_df(lda_data['model'],
                             lda_data['dict'],
                             lda_data['texts'],
                             lda_data['ids'])
out2.to_csv(top_distr_save_path)
