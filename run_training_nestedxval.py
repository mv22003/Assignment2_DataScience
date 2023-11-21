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
from sklearn.svm import SVC, LinearSVC
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
print('Number of jobs:', effective_n_jobs(-1))

# Cross-Validation
from sklearn.model_selection import GroupKFold, LeavePGroupsOut, StratifiedGroupKFold

# Prediction Scoring
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score

np.random.seed(3504)


def load_data(group):
    '''
    Loads data for the chosen group
    '''

    assert group in ['control', 'test', 'all'], "Invalid choice for 'group'. Must be ['control', 'test', 'all']"
    if group == 'control':
        df = pd.read_csv('./data/Final_Features_control.csv')
    elif group == 'test':
        df = pd.read_csv('./data/Final_Features_test.csv')
    else:
        df = pd.read_csv('./data/Final_Features_control.csv')
        df = pd.concat([df, pd.read_csv('./data/Final_Features_test.csv')])
    label = []
    th1, th2, th3 = 90, 97, 104
    for i in df['Empathy Score'].values:
        if i < th1:
            label.append(0)
        elif i < th2:
            label.append(1)
        elif i < th3:
            label.append(2)
        else:
            label.append(3)
    df['Empathy Score'] = label
    return df


def main(group, estimators):
    df = load_data(group)
    # Split into train-test sets, using subject id to leave 10% of subjects out
    lpgo = LeavePGroupsOut(n_groups=1)  # leave-one-subject-out cross-validation
    inner_cv = LeavePGroupsOut(n_groups=3)  # 10% of participants out
    # Number of possible splits:
    nsplits = lpgo.get_n_splits(groups=df['Participant name'])
    for name_clf, clf_params in estimators:
        print('Starting... %s' % name_clf)
        cv_results_out = []
        for out_i, (train_index, test_index) in enumerate(lpgo.split(df, groups=df['Participant name'])):
            cv_results = {}
            print('External cross-validation loop: %d/%d' % (out_i+1, nsplits))
            x_crossval = df.iloc[train_index, 1:-1].values  # participant ID is first column and label is last column
            y_crossval = df.iloc[train_index, -1].values
            groups_crossval = df.iloc[train_index, 0].values

            n_features = x_crossval.shape[-1]

            x_test = df.iloc[test_index, 1:-1].values  # one participant out
            y_test = df.iloc[test_index, -1].values
            param_grid = clf_params
            # Bayes Search with Cross-validation
            start = timeit.default_timer()
            if name_clf in ['SVC_linear', 'SVC_rbf']:  # Can't do RFE with these two because they don't have feature_importances_
                pipeline = Pipeline([('std', StandardScaler()), ('clf', clf_params['clf'])])
            else:
                param_grid['rfe__n_features_to_select'] = Integer(1, n_features, prior='uniform')
                pipeline = Pipeline([('std', StandardScaler()), ('rfe', RFE(estimator=clf_params['clf'][0])),
                                     ('clf', clf_params['clf'])])
            cv_clf = BayesSearchCV(pipeline, param_grid, n_iter=100, cv=inner_cv,  # 10% out
                                   scoring='balanced_accuracy',
                                   refit=True,  # it fits with the best estimator at the end
                                   n_jobs=effective_n_jobs(-1),
                                   n_points=max(1, effective_n_jobs(-1)),
                                   pre_dispatch=2 * effective_n_jobs(-1),
                                   return_train_score=True)
            cv_clf.fit(x_crossval, y_crossval, groups=groups_crossval)
            duration = timeit.default_timer() - start

            cv_results['crossval_train'] = [cv_clf.cv_results_["mean_train_score"][cv_clf.best_index_],
                                            cv_clf.cv_results_['std_train_score'][cv_clf.best_index_]]
            cv_results['crossval_val'] = [cv_clf.cv_results_["mean_test_score"][cv_clf.best_index_],
                                          cv_clf.cv_results_["std_test_score"][cv_clf.best_index_]]
            cv_results["best_estimator"] = cv_clf.best_estimator_
            cv_results["best_params"] = cv_clf.cv_results_["params"][cv_clf.best_index_]

            _ = plot_objective(cv_clf.optimizer_results_[0])
            optim_filepath = "./results_nested/optimizer_{}_{}_{}.png".format(group, name_clf, out_i+1)
            plt.savefig(optim_filepath)
            # Best model on test set
            y_pred = cv_clf.predict(x_test)
            cv_results['optim_test'] = balanced_accuracy_score(y_test, y_pred)
            cv_results['y_test'] = y_test
            cv_results['y_test_pred'] = y_pred

            print("\tOptimisation for {} done in {} seconds".format(name_clf, int(duration)))
            print("\t\tCrossval train results: {}+-{}".format(cv_results['crossval_train'][0],
                                                              cv_results['crossval_train'][1]))
            print("\t\tCrossval val results: {}+-{}".format(cv_results['crossval_val'][0],
                                                            cv_results['crossval_val'][1]))
            # Note: print below is wrong!! (Not changing it because that's how the code was executed.)
            # Pickled results are ok, but logs are not
            print("\t\tCrossval test set results: {}+-{}".format(cv_results['crossval_val'][0],
                                                                 cv_results['crossval_val'][1]))

            cv_results_out.append(cv_results)

            # Save at end of each one so we don't lose all results if it fails
            with open('./results_nested/cv_results_%s_%s.pkl' % (group, name_clf), 'wb') as handle:
                pickle.dump(cv_results_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


if __name__ == "__main__":
    # Estimator options
    params_clfs = [
        #('dummyClassifier', {'clf': [DummyClassifier()], 'clf__strategy': Categorical(['prior'])}),
        ('randomForest', {'clf': [RandomForestClassifier()], 'clf__n_estimators': Integer(50, 300, prior='uniform')}),
        ('adaboost', {'clf': [AdaBoostClassifier()], 'clf__n_estimators': Integer(50, 200, prior='uniform')}),
        ('LogisticRegression', {'clf': [LogisticRegression()], 'clf__C': Real(1e-5, 10, prior='log-uniform'),
                                'clf__penalty': Categorical([None, 'l1', 'l2']),
                                'clf__solver': Categorical(['saga'])}),
        ('SVC_linear', {'clf': [LinearSVC()], 'clf__C': Real(1e-5, 1000, prior='log-uniform')}),
        ('SVC_rbf', {'clf': [SVC()], 'clf__C': Real(1e-5, 1000, prior='uniform'), 'clf__kernel': ['rbf'],
                     'clf__gamma': Real(0.0001, 0.9, prior='log-uniform')}),
        ('XGBoost', {'clf': [XGBClassifier()],
                     'clf__max_depth': Integer(3, 15, prior='uniform'),
                     'clf__n_estimators': Integer(100, 500, prior='uniform'),
                     'clf__reg_alpha': Real(0.0001, 10, prior='log-uniform'),  # l1 reg
                     'clf__reg_lambda': Real(0.01, 100, prior='log-uniform'),  # l2 reg
                     'clf__learning_rate': Real(5e-4, 0.5, prior='log-uniform'),
                     'clf__subsample': Real(0.3, 1, prior='uniform')}),
        #('LGBMClassifier', {'clf': [LGBMClassifier()],
        #                    'clf__learning_rate': Real(5e-4, 0.5, prior='log-uniform'),
        #                    'clf__subsample': Real(0.3, 1, prior='uniform'),
        #                    'clf__n_estimators': Integer(100, 500, prior='uniform'),
        #                    'clf__max_depth': Integer(3, 15, prior='uniform'),
        #                    'clf__num_leaves': Integer(5, 40, prior='uniform'),
        #                    'clf__reg_alpha': Real(0.0001, 10, prior='log-uniform'),  # l1 reg
        #                    'clf__reg_lambda': Real(0.01, 100, prior='log-uniform')  # l2 reg
        #                    })
    ]
    for group in ['control', 'test']:#, 'all']:
        _ = main(group, params_clfs)
