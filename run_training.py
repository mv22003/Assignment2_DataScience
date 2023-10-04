# Ignore warnings
import warnings

warnings.simplefilter('ignore')

# Basic libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from skopt.plots import plot_objective
import timeit

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Optimization
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV  # pip install scikit-optimize
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from skopt.space import Real, Categorical, Integer
from joblib import effective_n_jobs

# Cross-Validation
from sklearn.model_selection import GroupKFold, LeavePGroupsOut
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid

# Prediction Scoring
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score

np.random.seed(3504)


def load_data(group):
    '''
    Loads data for the chosen group
    '''

    assert group in ['control', 'test', 'all'], "Invalid choice for 'group'. Must be ['control', 'test', 'all']"
    if group == 'control':
        df = pd.read_csv('Final_Features_control.csv')
    elif group == 'test':
        df = pd.read_csv('Final_Features_test.csv')
    else:
        df = pd.read_csv('Final_Features_control.csv')
        df = pd.concat([df, pd.read_csv('Final_Features_test.csv')])
    return df


def main(group, estimators):
    df = load_data(group)
    # Split into train-test sets, using subject id to leave 10% of subjects out
    lpgo = LeavePGroupsOut(n_groups=6 if group == 'all' else 3)
    # Number of possible splits:
    nsplits = lpgo.get_n_splits(groups=df['Participant name'])
    # Pick a random one - this is our train/test split for the rest of the programme.
    # The training set will be used for cross-val and hyperparameter tuning
    split = np.random.randint(0, nsplits)
    for i, (train_index, test_index) in enumerate(lpgo.split(df, groups=df['Participant name'])):
        if i == split:
            break

    x_crossval = df.iloc[train_index, 1:-1]  # assuming participant ID is first column and label is last column
    y_crossval = df.iloc[train_index, -1]
    groups_crossval = df.iloc[train_index, 0]

    n_features = x_crossval.shape[-1]

    x_test = df.iloc[test_index, 1:-1]
    y_test = df.iloc[test_index, -1]

    inner_cv = GroupKFold(n_splits=9)
    # for 'test' and 'control', we have 27 participants in the crossval group, so we leave 3 participants out each time
    # for 'all', we have 54 in cross-val, so we leave 6 participants out each time

    cv_results = {}
    for name_clf, clf_params in estimators:
        print('Starting... %s' % name_clf)
        cv_results[name_clf] = {}
        param_grid = clf_params
        param_grid['rfe__n_features_to_select'] = Integer(1, n_features, prior='uniform')

        # Bayes Search with Cross-validation
        start = timeit.default_timer()
        pipeline = Pipeline([('std', StandardScaler()), ('rfe', RFE(estimator=clf_params['clf'])),
                             ('clf', clf_params['clf'])])
        cv_clf = BayesSearchCV(pipeline, clf_params, n_iter=100, cv=inner_cv,
                               scoring='balanced_accuracy',
                               refit=True,  # it fits with the best estimator at the end
                               n_jobs=effective_n_jobs(-1),
                               n_points=max(1, effective_n_jobs(-1)),
                               pre_dispatch=2 * effective_n_jobs(-1),
                               return_train_score=True)
        cv_clf.fit(x_crossval, y_crossval, groups=groups_crossval)
        duration = timeit.default_timer() - start

        cv_results[name_clf]['crossval_train'] = [cv_clf.cv_results_["mean_train_score"][cv_clf.best_index_],
                                                  cv_clf.cv_results_['std_train_score'][cv_clf.best_index_]]
        cv_results[name_clf]['crossval_val'] = [cv_clf.cv_results_["mean_test_score"][cv_clf.best_index_],
                                                cv_clf.cv_results_["std_test_score"][cv_clf.best_index_]]
        cv_results["best_estimator"] = cv_clf.best_estimator_
        cv_results["best_params"] = cv_clf.cv_results_["params"][cv_clf.best_index_]

        _ = plot_objective(cv_clf.optimizer_results_[0])
        optim_filepath = "./results/optimizer_{}_{}.png".format(group, name_clf)
        plt.savefig(optim_filepath)
        # Best model on test set
        y_pred = cv_clf.predict(x_test)
        cv_results[name_clf]['optim_test'] = balanced_accuracy_score(y_test, y_pred)

        print("\tOptimisation for {} done in {} seconds".format(name_clf, int(duration)))
        print("\t\tCrossval train results: {}+-{}".format(cv_results[name_clf]['crossval_train'][0],
                                                          cv_results[name_clf]['crossval_train'][1]))
        print("\t\tCrossval test results (validation sets): {}+-{}".format(cv_results[name_clf]['crossval_val'][0],
                                                                           cv_results[name_clf]['crossval_val'][1]))

        # Save at end of each one so we don't lose all results if it fails
        with open('./results/cv_results_%s.pickle' % group, 'wb') as handle:
            pickle.dump(cv_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


if __name__ == "__main__":
    # Estimator options
    params_clfs = [
        ('dummyClassifier', {'clf': [DummyClassifier()], 'clf__strategy': Categorical(['prior'])}),
        ('randomForest', {'clf': [RandomForestClassifier()], 'clf__n_estimators': Integer(50, 300, prior='uniform')}),
        ('decisionTree', {'clf': [DecisionTreeClassifier()], 'clf__criterion': Categorical(['entropy']),
                          'clf__max_depth': Integer(1, 10, prior='uniform')}),
        ('adaboost', {'clf': [AdaBoostClassifier()], 'clf__n_estimators': Integer(50, 200, prior='uniform')}),
        ('LogisticRegression', {'clf': [LogisticRegression()], 'clf__C': Real(1e-5, 10, prior='log-uniform'),
                                'clf__penalty': Categorical([None, 'l1', 'l2']),
                                'clf__solver': Categorical(['saga'])}),
        ('SVC_rbf', {'clf': [SVC()], 'clf__C': Real(1e-5, 1000, prior='uniform'), 'clf__kernel': ['rbf'],
                     'clf__gamma': Real(0.0001, 0.9, prior='log-uniform')}),
        ('SVC_linear', {'clf': [SVC()], 'clf__C': Real(1e-5, 1000, prior='log-uniform'),
                        'clf__kernel': Categorical(['linear'])}),
        ('XGBoost', {'clf': [XGBClassifier()], 'clf__max_depth': Integer(1, 20, prior='uniform'),
                     'clf__reg_alpha': Real(0.0001, 10, prior='log-uniform'),
                     'clf__reg_lambda': Real(0.01, 100, prior='log-uniform'),
                     'clf__learning_rate': Real(5e-5, 0.05, prior='log-uniform')}),
        ('sgd', {'clf': [SGDClassifier()]}),
        ('kNeighbors', {'clf': [KNeighborsClassifier()], 'clf__n_neighbors': Integer(3, 10, prior='uniform')}),
        ('nearestCentroid', {'clf': [NearestCentroid()]}),
        ('LGBMClassifier', {'clf': [LGBMClassifier()], 'clf__learning_rate': Real(0.0005, 0.3, prior='log-uniform'),
                            'clf__subsample': Real(0.3, 1, prior='uniform'),
                            'clf__n_estimators': Integer(500, 1000, prior='uniform'),
                            'clf__max_depth': Integer(3, 15, prior='uniform'),
                            'clf__num_leaves': Integer(5, 40, prior='uniform'),
                            'clf__reg_alpha': Real(0.0001, 10, prior='log-uniform'),
                            'clf__reg_lambda': Real(0.01, 100, prior='log-uniform')})
    ]
    for group in ['control', 'test', 'all']:
        _ = main(group, params_clfs)
