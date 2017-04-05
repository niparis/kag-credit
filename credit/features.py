import logging
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn import preprocessing

log = logging.getLogger(__name__)


def encode_category(*, df, col_name, bins):
    """ binarization of a categorical variable
    """
    encoder = preprocessing.OneHotEncoder()
    label_encoder = preprocessing.LabelEncoder()
    # dataframe labels
    col_labels = []
    for x, y in zip(bins, bins[1:]):
        col_labels.append('{}-{}_{}'.format(col_name, x, y))

    df['tcol'] = label_encoder.fit_transform(df[col_name])
    age_cat_data = encoder.fit_transform(df[['tcol']].as_matrix()).toarray()
    age_cat_df = pd.DataFrame(age_cat_data, columns=col_labels)
    df = pd.concat([df, age_cat_df], axis=1)
    return df.drop(labels=[col_name, 'tcol'], axis='columns')


def pre_process(*, df) -> pd.DataFrame:
    """ Preprocessing function
        Returns: Pandas DataFrame

        Notes: Graveyard - ideas that have been tried but are not helping

            - many kids at a young age seems dangerous
            df['age_minus_deps'] = df['age'] - ((df['NumberOfDependents'] + 1) * 10)
            - income per dependant >> fails
            df['inc-per-dep'] = df['MonthlyIncome'] / df['NumberOfDependents']
            - idea : a high debt ratio is more dangerous when we have a negative cash out >> fails
            df['debt-ratio-impact'] = df['cash-out'] * df['DebtRatio']
            -
            df['misery-product'] = df['sum_debt_weight'] * (df['all_days_late'] / 10)
            - capping of days late column >> fails
            for col_name in three_pastdue_cols:
                df.loc[df[col_name] >= 10, col_name] = 10
            - threshold on the number of days late [1, 3, 10]
                df.loc[df['all_days_late'] > 3, 'many_late_days'] = 1
                df.loc[df['all_days_late'] <= 3, 'many_late_days'] = 0
            - new  feature "special 98" >> fails
                extra_feature = 'numbertimenotworse-special90+'
                df.loc[df['NumberOfTime30-59DaysPastDueNotWorse'].isin([96, 98]), extra_feature] = 1
                df.loc[df[extra_feature] != 1, extra_feature] = 0
            - capping credit_lines_cols >> negative impact
                for col_name in credit_lines_cols:
                    df.loc[df[col_name] >= 3, col_name] = 3
            - room_for_maneuver: scaled product of RevolvingUtilizationOfUnsecuredLines and all_loans. Neat idea, doesnt work
                scaler = preprocessing.StandardScaler()
                bob = scaler.fit_transform(df[['RevolvingUtilizationOfUnsecuredLines', 'all_loans']])
                df['scale+prod+man+room'] = pd.DataFrame(bob[:, 0] * bob[:, 1], columns=['scale+prod+man+room']).fillna(0).astype(int)
                df['room_for_maneuver'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['all_loans']
    """

    ### AGE ###
    df.loc[df['age'] == 0, 'age'] = df['age'].median()
    step = 10
    bins = [x for x in range(20, 115, step)]
    df['age-bin'] = pd.cut(df['age'], bins)
    # binarization of age-bins
    df = encode_category(df=df, col_name='age-bin', bins=bins)
    # addding a retired category
    df.loc[(df['age'] >= 60) & (df['age'] <= 99), 'retired_2digitage'] = 1
    df.loc[df['retired_2digitage'] != 1, 'retired_2digitage'] = 0

    # young with kids is more dangerous >> shows as 0 weight but give better results
    df.loc[(df.age < 35) & (df.NumberOfDependents > 2), 'young_w_kids'] = 1
    df.loc[(df.young_w_kids != 1), 'young_w_kids'] = 0

    ###  PAST DUE ####
    three_pastdue_cols = ['NumberOfTime30-59DaysPastDueNotWorse',
                          'NumberOfTime60-89DaysPastDueNotWorse',
                          'NumberOfTimes90DaysLate']
    # all days of lateness
    df['all_days_late'] = df[three_pastdue_cols].sum(axis=1)

    ### CREDIT LINES ###
    credit_lines_cols = ['NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']
    df['all_loans'] = df[credit_lines_cols].sum(axis=1)

    ### RevolvingUtilizationOfUnsecuredLines ###
    col = 'RevolvingUtilizationOfUnsecuredLines'

    # proposed solution : divide by 1000 all values greater than 100
    df.loc[df[col] >= 100, col] = df.loc[df[col] >= 100, col] / 1000
    df.loc[df[col] >= 10, col] = df.loc[df[col] >= 10, col] / 10  # done on purpose.
    # binning
    bins = [-1, 0, 0.5, 1, 1.2, 1.5, 2, 5, 10]
    df['rev-unsec'] = pd.cut(df[col], bins)
    ## if i have a high RevolvingUtilizationOfUnsecuredLines, but few loans, i can probably just take more loans
    df.loc[(df[col] > 0.75) & (df['all_loans'] > 3), 'can_take_loans'] = 0
    df.loc[df.can_take_loans != 0, 'can_take_loans'] = 1

    # synthetic feature
    df['sum_debt_weight'] = df['RevolvingUtilizationOfUnsecuredLines'] + df['DebtRatio']

    # early warning system
    df.loc[df['DebtRatio'] > 0.6, 'high_indebt_0.6+'] = 1
    df.loc[df['DebtRatio'] <= 0.6, 'high_indebt_0.6+'] = 0

    # very dangerous
    df.loc[df['DebtRatio'] > 0.9, 'high_indebt_0.9+'] = 1
    df.loc[df['DebtRatio'] <= 0.9, 'high_indebt_0.9+'] = 0

    # binarization of rev-unsec
    df = encode_category(df=df, col_name='rev-unsec', bins=bins)

    ### Debt Ratio & income ###
    col = 'DebtRatio'
    bins = [-1, 0, 0.5, 0.8, 1, 1.25, 1.5, 2, 10, 100, 1000, 10000, 1000000]
    df['debt-ratio-bin'] = pd.cut(df[col], bins)
    df = encode_category(df=df, col_name='debt-ratio-bin', bins=bins)

    # now normalizing the Debt Ratio
    # Why ? make it usable as a continuous var + better computation of expenses
    # ugly but no time
    df.loc[pd.isnull(df[col]), col] = df[col].median()
    df.loc[(df[col] >= 100000), col] = df.loc[(df[col] >= 100000), col] / 100000
    df.loc[(df[col] >= 10000), col] = df.loc[(df[col] >= 10000), col] / 10000
    df.loc[(df[col] >= 1000), col] = df.loc[(df[col] >= 1000), col] / 1000
    df.loc[(df[col] >= 100), col] = df.loc[(df[col] >= 100), col] / 100
    df.loc[(df[col] >= 4), col] = df.loc[(df[col] >= 4), col] / 10

    # income
    df.loc[pd.isnull(df['MonthlyIncome']), 'MonthlyIncome'] = df['MonthlyIncome'].median()

    # computing expenses
    df['monthly-expenses'] = df['MonthlyIncome'] * df['DebtRatio']
    # cash out
    df['cash-out'] = df['MonthlyIncome'] - df['monthly-expenses']
    # binning the cash outflow
    bins = [-np.inf, -10000, -100, 0, 2500, 5000, 10000, np.inf]
    df['cash-out-bin'] = pd.cut(df['cash-out'], bins)
    df = encode_category(df=df, col_name='cash-out-bin', bins=bins)
    # to drop
    col_removal = ['cash-out-bin--100_0', 'cash-out-bin--inf_-10000', 'cash-out-bin-2500_5000',
                   'age-bin-20_30', 'debt-ratio-bin-10000_1000000', 'debt-ratio-bin-1000_10000',
                   'age-bin-80_90', 'debt-ratio-bin-2_10', 'debt-ratio-bin-1.5_2', 'debt-ratio-bin-1.25_1.5',
                   'debt-ratio-bin-1_1.25', 'age-bin-90_100', 'debt-ratio-bin--1_0', 'rev-unsec-5_10',
                   'rev-unsec-2_5', 'rev-unsec-1.5_2', 'rev-unsec-1.2_1.5',
                   'age-bin-100_110', 'debt-ratio-bin-0.8_1', 'cash-out-bin-10000_inf', 'debt-ratio-bin-0.5_0.8',
                   'debt-ratio-bin-10_100', 'age-bin-30_40', 'age-bin-70_80']

    df = df.drop(labels=col_removal, axis='columns', errors='ignore')
    # keep only relevant features
    log.info('Columns left : {}'.format(', '.join(df.columns)))
    # import ipdb ; ipdb.set_trace()
    return df.fillna(0)  # can't afford to have NaNs after this step


def scale_features(*, df, scale=True) -> Tuple[np.ndarray, np.ndarray, list]:
    """ Place to select the columns we choose to train the model
        Note: scaling is basically useless with ensemble algos.
        Returns: tuple of np.array + list with feature names
    """
    # Scale
    if scale:
        scale_cols = ['MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                      'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
                      'age']
        scaler = preprocessing.StandardScaler()
        log.info('Scaling following columns : %s', ', '.join(scale_cols))
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        log.info('Not scaling')

    # Extract arrays
    y = df['SeriousDlqin2yrs'].values
    X = df.drop('SeriousDlqin2yrs', axis='columns').values
    feature_names = df.drop('SeriousDlqin2yrs', axis='columns').columns.tolist()

    log.info(X.shape)
    log.info(y.shape)

    return X, y, feature_names
