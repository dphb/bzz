# IMPORTS #

import logging
import warnings

import dill as pickle
import pandas as pd

import params_lda as par
from functions_lda import generate_outputs

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# %%
# LOAD MODEL SPECS #

with open(par.LDA_PICKLE_PATH, 'rb') as f:
    lda_data = pickle.load(f)

topic_df, topic_distr_df = generate_outputs(lda_data['model'],
                                            lda_data['df'],
                                            lda_data['dict'],
                                            save_output=True)

# topic_df.to_csv('topics.csv', index=False)
# topic_distr_df.to_csv('topic_distr_df.csv', index=False)
