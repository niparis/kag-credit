import logging
import os
import datetime as dt

import pandas as pd

logging.basicConfig(level=logging.INFO)

__this_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.abspath(os.path.join(__this_dir, '..', 'data'))
output_root = os.path.abspath(os.path.join(__this_dir, '..', 'output'))


def load_test() -> pd.DataFrame:
    """ Loads and return the test source data, as a Pandas DataFrame
    """
    return (pd.read_csv(os.path.join(data_root, 'cs-test.csv'))
              .drop(labels=['Unnamed: 0'], axis='columns')
            )


def load_training() -> pd.DataFrame:
    """ Loads and return the training source data, as a Pandas DataFrame
    """
    return (pd.read_csv(os.path.join(data_root, 'cs-training.csv'))
              .drop(labels=['Unnamed: 0'], axis='columns')
            )


def get_model_fname(estimator) -> str:
    """ returns full path to the file name of a the trained model
    """
    fname = '%s.model' % estimator.__class__.__name__
    return os.path.join(output_root, fname)


def get_output_fname(estimator) -> str:
    """ returns full path to the file name of an output file
    """
    fname = '%s-%s.csv' % (estimator.__class__.__name__, int(dt.datetime.utcnow().timestamp()))
    return os.path.join(output_root, fname)
