import os
import warnings

import click
import xgboost as xgb
from sklearn import ensemble, linear_model, naive_bayes, svm

import credit

# Possible fix for multiprocessing problems with RandomForestClassifier
# https://github.com/scikit-learn/scikit-learn/issues/4459
# import multiprocessing
# affinity.set_process_affinity_mask(0, 2**multiprocessing.cpu_count()-1)
# http://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
warnings.filterwarnings("ignore")


@click.group()
def cli():
    """Kaggle CLI.

    Order of commands :\n
        1] train: Trains a model, outputs metrics, saves the model to disk. \n
        2] predict: Saves prediction file \n
        3] clean: deletes all preductions files (not the models). \n
    """
    pass


@click.command()
def clean():
    """ Cleans all output files
    """
    c = 0
    for path, folders, files in os.walk(credit.base.output_root):
        for_deletion = [f for f in files if not f.endswith('.model')]
        for f in for_deletion:
            file_path = os.path.join(path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
                c += 1
    click.echo('%s output files deleted' % c)


""" Algos
    Defining algos with their parameters
"""

# parameters to try for hyperparams optimization
HYPER_OPTI = {
    'log': {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
            'C': [0.5, 1, 5, 10, 100, 1000]},
    'svc': [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}],
    'rndforest': {
        'min_samples_leaf': [10, 100, 1000],
        "max_depth": [3, None],
        'min_weight_fraction_leaf': [0.001, 0.01, 0.1, 0.5],
        "min_samples_split": [0.5, 2, 4, 5, 7, 8],
        'max_features': ['sqrt', 'log2', None, 0.2, 0.7],
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False]
    },
    'xgb': {
        'max_depth': [3],
        'min_child_weight': [1],
        'gamma': [i/10.0 for i in range(0,5)]
    }
}

# Algos : values are tuple of: estimator class, stable paramters, optimized paramters
ALGOS = {
    'xgb': (xgb.XGBClassifier,
            {
                'learning_rate': 0.025,
                'n_estimators': 600,
                'seed': 0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                # Control the balance of positive and negative weights, useful for unbalanced classes.
                # A typical value to consider: sum(negative cases) / sum(positive cases)
                'scale_pos_weight': 0.066,
            },
            {
                'max_depth': 7,
                'min_child_weight': 20,
                'gamma': 0.33,
            }
            ),
    'rndforest': (ensemble.RandomForestClassifier,
            {
                'n_estimators': 100,

            },
            {
                'min_weight_fraction_leaf': 0.001,
            }
                ),
    'log': (linear_model.LogisticRegression,
            {'max_iter': 100, 'penalty': 'l2', 'dual': False},
            {'solver': 'liblinear', 'C': 0.5}
            ),
    'gaussian': (naive_bayes.GaussianNB,
                 {},
                 {}
                 ),
}


@click.option('--hyper', is_flag=True, default=False)
@click.option('--jobs', default=1, help='How many CPU cores to use')
@click.argument('algo')
@click.command()
def train(algo, hyper=None, fast=None, jobs=None):
    """ Trains the model. Print its result (AUC)
    Saves the model
    """
    estimator, params, opti_params = ALGOS.get(algo)

    if estimator is not None:
        params.update(opti_params)

        if algo not in ['xgb']:
            params['n_jobs'] = int(jobs)

        estimator_instance = estimator(**params)
        if hyper:
            hyper_params = HYPER_OPTI.get(algo)
            if hyper_params:
                credit.model.hyper(estimator_instance, algo=algo,
                                   hyperopti=hyper_params, n_jobs=jobs)
            else:
                click.echo('No hyper params found for tuning (in HYPER_OPTI global)')
        else:
            credit.model.train(estimator_instance, opti_params=opti_params, algo=algo, n_jobs=jobs)
    else:
        click.echo('Existing estimators: %s' % ', '.join(ALGOS.keys()))


@click.argument('algo')
@click.command()
def predict(algo):
    """ Saves the prediction (output file)
    """
    estimator, _, _ = ALGOS.get(algo)
    if estimator is not None:
        credit.model.predict(estimator=estimator())
    else:
        click.echo('Existing estimators: %s' % ', '.join(ALGOS.keys()))


cli.add_command(clean)
cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
