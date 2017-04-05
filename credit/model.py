import logging

from sklearn import model_selection
from sklearn.externals import joblib
import pandas as pd
import xgboost as xgb

from . import base
from . import features
from . import metrics

log = logging.getLogger(__name__)


def hyper(estimator, algo, hyperopti=None, n_jobs=None):
    """ Hyper params optimization of a model
        Does not save anything.
        User is expected to copy the selected parameters in the ALGOS global
    """

    log.info('Starting training of %s', estimator.__class__.__name__)
    log.info('n_jobs: %s', n_jobs)
    training_df = base.load_training()

    # This is the key
    preproc_df = features.pre_process(df=training_df)

    # salect, scale & get the np.array we need
    X, y, _ = features.scale_features(df=preproc_df, scale=False)

    # hyper params optimization
    gridsearch = model_selection.GridSearchCV(estimator,
                                              hyperopti,
                                              cv=5,
                                              verbose=3,
                                              n_jobs=n_jobs,
                                              iid=False,
                                              scoring='auc')
    # fit model
    gridsearch.fit(X, y)
    log.info("Grid scores")
    log.info(gridsearch.grid_scores_)
    log.info("Best parameters set found on development set:")
    log.info(gridsearch.best_params_)
    log.info("Best Score")
    log.info(gridsearch.best_score_)


def train(estimator, algo, opti_params=None, hhyperopti=None, n_jobs=None):
    """ Train a model
    """

    log.info('Starting training of %s', estimator.__class__.__name__)
    log.info('n_jobs: %s', n_jobs)
    training_df = base.load_training()

    # This is the key
    preproc_df = features.pre_process(df=training_df)

    # salect, scale & get the np.array we need
    X, y, feature_names = features.scale_features(df=preproc_df, scale=True)

    # cross_validation
    cv = model_selection.ShuffleSplit(n_splits=5,
                                      test_size=0.2,
                                      random_state=0
                                      )

    log.info('Cross Validation')
    if algo == 'xgb':
        dtrain = xgb.DMatrix(X, label=y)
        results = xgb.cv(opti_params,
                         dtrain,
                         num_boost_round=20,
                         nfold=5,
                         metrics='auc',
                         seed=0,
                         callbacks=[xgb.callback.print_evaluation(show_stdv=True)]
                         )
        log.info(results.mean())
    else:
        scores = model_selection.cross_val_score(estimator,
                                                 X,
                                                 y,
                                                 cv=cv,
                                                 scoring='roc_auc',
                                                 n_jobs=n_jobs
                                                 )
        log.info('CV performance:')
        log.info("AUC: {:.2f} % (+/- {:.2f} )".format(scores.mean() * 100,
                                                      scores.std() * 2 * 100
                                                      )
                 )
    # fit model
    log.info('Fitting model')
    if algo == 'xgb':
        estimator.fit(X, y, eval_metric='auc')
        # importance
        imp = (pd.DataFrame(estimator.feature_importances_,
                            index=feature_names,
                            columns=['importance'])
                 .sort_values(by='importance', ascending=False)
               )
        log.info(imp)

    else:
        estimator.fit(X, y)

    # saving
    path = base.get_model_fname(estimator)
    joblib.dump(estimator, path)

    # predict
    y_pred = estimator.predict(X)

    # show results
    y_score = estimator.predict_proba(X)[:, 1]
    metrics.simple_metrics(y_true=y, y_pred=y_pred, y_score=y_score)
    log.info(estimator)


def predict(*, estimator) -> None:
    """ Creates a prediction file from a SAVED model
    """
    # loading
    path = base.get_model_fname(estimator)
    estimator = joblib.load(path)

    test_df = base.load_test()
    # This is the key
    preproc_df = features.pre_process(df=test_df)

    # salect, scale & get the np.array we need
    X, y, _ = features.scale_features(df=preproc_df, scale=True)

    y_score = estimator.predict_proba(X)[:, 1]

    # saving
    path = base.get_output_fname(estimator)
    df = pd.DataFrame(y_score).reset_index()
    df.columns = ['Id', 'Probability']
    df['Id'] += 1
    df.to_csv(path, index=None)
    log.info('Saved at %s' % path)
    log.info(estimator)
