# -*- coding: utf-8 -*-
import sys
from sklearn import linear_model as sklm
from sklearn import tree as sktree
from sklearn import cross_validation as cv
from sklearn.metrics import roc_curve, auc, make_scorer
from sklearn import preprocessing as prepro
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn import ensemble as en
from numpy import argsort
import xgboost as xgb
import numpy as np
#import pylab as pl
import logging
import datetime
import os
import pprint as pp
import settings
import pickle
import scipy


def basic_train_test_split(np_data, y):
    logger = logging.getLogger('model-learner.model_building.basic_train_test_split')
    X_train, X_test, y_train, y_test = cv.train_test_split(np_data, y, test_size=0.5, random_state=0)

    return X_train, X_test, y_train, y_test

def kfold_training_set_split(np_data, y, n_folds=10):
    # Use stratified k fold sampling to create training and test set.
    skf = cv.StratifiedKFold(y, n_folds=n_folds)
    # for train_index, test_index in skf:
    #     X_train, X_test = np_data[train_index], np_data[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    # return X_train,y_train,X_test,y_test

    return skf

def logistic_regression(X_train, y_train, X_test, y_test, classifiers, label, \
                        RESULTS_OUTPUT_DIR, MODELS_OUTPUT_DIR, pickle=False, **kwargs):
    logger = logging.getLogger('model-learner.model_building.logistic_regression')

    model, classifiers = calc_clf_model(sklm.LogisticRegression(**kwargs), X_train, y_train, \
                                    X_test, y_test, label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)

    logger.info('Started Log Reg 5 model build %s', datetime.datetime.now().time().isoformat())
    logger.info('Model coefficients: %s', pp.pformat(model.coef_))
    logger.info('Model intercept: %s', pp.pformat(model.intercept_))

    return classifiers

def random_forest(X_train, y_train, X_test, y_test, classifiers, \
                 label, RESULTS_OUTPUT_DIR, MODELS_OUTPUT_DIR, pickle=False, **kwargs):
    logger = logging.getLogger('model-learner.model_building.random_forest')

    model, classifiers = calc_clf_model(en.RandomForestClassifier(**kwargs), X_train, y_train, X_test, y_test, \
                                        label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)

    return classifiers

def decision_tree(X_train, y_train, X_test, y_test, classifiers, label, \
                RESULTS_OUTPUT_DIR, MODELS_OUTPUT_DIR, pickle=False, **kwargs):
    logger = logging.getLogger('model-learner.model_building.decision_tree')

    model, classifiers = calc_clf_model(sktree.DecisionTreeClassifier(**kwargs), X_train, y_train, \
                                    X_test, y_test, label, classifiers, MODELS_OUTPUT_DIR, pickle=pickle)

    return classifiers

def logistic_regression_GridSearchCV(param_grid, X_train, y_train, X_test, y_test, classifiers, model_id \
                                ,RESULTS_OUTPUT_DIR, MODELS_OUTPUT_DIR, score='accuracy', pickle=False):
    logger = logging.getLogger('model-learner.model_building.logistic_regression_GridSearch')

    # set the estimator
    model_estimator= sklm.LogisticRegression(class_weight='auto')
    # set ks as cost function
    score_ks = make_scorer(cost_function_ks, greater_is_better=True, needs_proba=True)
    # set GridSearchCV as cross validation
    gd_clf = GridSearchCV(model_estimator, param_grid, n_jobs=1, refit=True, cv=3, verbose=1, scoring=score_ks)

    # there are two loop to run cross validation and param selection iterations
    classifiers = run_GridSearchCV_fit(gd_clf, X_train, y_train, X_test, y_test, model_id, classifiers, MODELS_OUTPUT_DIR, pickle)

    return classifiers

def run_GridSearchCV_fit(gd_clf, X_train, y_train, X_test, y_test, model_id, classifiers, MODELS_OUTPUT_DIR, pickle):
    logger = logging.getLogger('model-learner.model_building.run_GridSearchCV_fit')

    # fit the model
    print('start build model.')
    model = gd_clf.fit(X_train, y_train)
    print "returnValue:",gd_clf.best_score_

    print('save GridSearchCV model.')
    if pickle:
        #name_list = joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl)
        name_list =  joblib.dump(model,model_id,compress = 3)
        #print name_list
    print('\n ==> model: ')
    #print('\n ==> grid scores: ')
    #print gd_clf.grid_scores_
    #print("\n ==> best ks score: %0.3f" % gd_clf.best_score_)
    #print('\n ==> best estimator: ')
    #print gd_clf.best_estimator_
    #print('\n ==> best param: ')
    #print gd_clf.best_params_
    best_parameters = dict();
    best_parameters = gd_clf.best_estimator_.get_params()
    save_plot_data(model, X_test, y_test, model_id)
    return classifiers

def xgboost(X_train,y_train, X_test, y_test,clf,label,model_saved_path,pickle,**sklearn_params):
    s_p = np.sum(y_train == 1) + np.sum(y_test == 1)
    s_n = np.sum(y_train == 0) + np.sum(y_test == 0)
    print X_train.shape,X_test.shape
    ratio = float(s_p) / np.sum(s_n)
    clf = xgb.XGBClassifier(missing=9999999999,
                            max_depth=3,
                            n_estimators=100,
                            learning_rate=0.1,
                            nthread=4,
                            subsample=1.0,
                            colsample_bytree=0.5,
                            min_child_weight=3,
                            scale_pos_weight=ratio,
                            seed=4242)

    clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
            eval_set=[(X_train, y_train), (X_test, y_test)])

    print('save xgboost model.')
    if pickle:
        #name_list = joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl)
        name_list =  joblib.dump(clf,model_saved_path,compress = 3)
        #print name_list
    print('\n ==> model: ')

    save_plot_data(clf, X_test, y_test, model_saved_path)
    return clf



def score_normalization_batch(proba_b):
    scores_fixed = []
    for prob in proba_b:
        score = 300 + (600 * prob)
        scores_fixed.append(score)
    return scores_fixed

def score_normalization(min, max, p):
    return min + (max * p)

def save_plot_data(model, X_test, y_test, label):
    y_proba = model.predict_proba(X_test)
    proba_w = [x[0] for x in y_proba]
    proba_b = [x[1] for x in y_proba]

    #dump to file
    plot_img_name_prefix = os.path.split(label)
    proba_file_name = plot_img_name_prefix[0] + "/proba_b.pkl"
    test_file_name = plot_img_name_prefix[0] + "/y_test.pkl"
    out_proba = open(proba_file_name, 'wb')
    out_test = open(test_file_name, 'wb')
    print proba_file_name,test_file_name
    pickle.dump(proba_b, out_proba)
    pickle.dump(y_test, out_test)


def cost_function_ks(ground_truth, predictions):
    # print predictions.shape
    # print predictions
    step = 1
    y_test = ground_truth

    proba_w = [x[0] for x in predictions]
    proba_b = [x[1] for x in predictions]

    ind_b = argsort(proba_b)
    ind_w = argsort(proba_w)
    # ind_w = argsort(proba_w)[::-1]
    plot_data_b = [0]
    plot_data_w = [0]

    b = sum(y_test)
    w = len(y_test) - b

    dif = 0

    # for
    print  b,w
    b = b + 1
    w = w + 1
    for i in range(step, 100+step, step):
        s_b = 0
        s_w = 0
        for i in range(len(y_test)*i/100):
            s_b += y_test[ind_b[i]]
            if y_test[ind_w[i]] == 0:
                s_w += 1

        plot_data_b.append(s_b * 100.0 / b)
        plot_data_w.append(s_w * 100.0 / w)

        temp = s_w * 100.0 / w - s_b * 100.0 / b
        if temp > dif:
            dif = temp

    #print "k-s value: ", dif

    return dif

def calc_clf_model(clf, X_train, y_train, X_test, y_test, label, classifiers, MODELS_OUTPUT_DIR, pickle=False):
    logger = logging.getLogger('model-learner.model_building.calc_clf_model')

    model = clf.fit(X_train, y_train)
    metrics = model_metrics(model, X_test, y_test, X_train, y_train, label)

    classifiers[label] = metrics

    if pickle:
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl',compress = 3)

    return model, classifiers

def model_metrics(model, X_test, y_test, X_train, y_train, label):
    logger = logging.getLogger('model-learner.model_building.model_metrics')

    test_fpr, test_tpr, test_roc_auc = calc_model_metrics(model, X_test, y_test)
    train_fpr, train_tpr, train_roc_auc = calc_model_metrics(model, X_train, y_train)

    print("area under ROC curve for model " + label + " test, train: %f" % test_roc_auc,train_roc_auc)

    # build a dictionary with the relevant output from the model
    outputDict = {
                'label':label,
                'test_false_positive':test_fpr,
                'test_true_positive':test_tpr,
                'test_ROC_auc':test_roc_auc,
                'train_false_positive':train_fpr,
                'train_true_positive':train_tpr,
                'train_ROC_auc':train_roc_auc
                }

    return outputDict

def calc_model_metrics(model, X, y):
    logger = logging.getLogger('model-learner.model_building.calc_model_metrics')

    test_probas_ = model.predict_proba(X)
    test_fpr, test_tpr, test_thresholds = roc_curve(y, test_probas_[:,1])
    test_roc_auc = auc(test_fpr, test_tpr)

    return test_fpr, test_tpr, test_roc_auc
