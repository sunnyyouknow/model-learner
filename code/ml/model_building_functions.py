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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='/usr/share/fonts/simhei.ttf')
import matplotlib.pyplot as plt
import pylab as pl
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

def logistic_regression_GridSearchCV(param_grid, X_train, y_train, X_test, y_test, classifiers, label \
                                ,RESULTS_OUTPUT_DIR, MODELS_OUTPUT_DIR, score='accuracy', pickle=False):
    logger = logging.getLogger('model-learner.model_building.logistic_regression_GridSearch')

    # set the estimator
    model_estimator= sklm.LogisticRegression(class_weight='auto')
    # set ks as cost function
    score_ks = make_scorer(cost_function_ks, greater_is_better=True, needs_proba=True)
    # set GridSearchCV as cross validation
    gd_clf = GridSearchCV(model_estimator, param_grid, n_jobs=1, refit=True, cv=3, verbose=1, scoring=score_ks)

    # there are two loop to run cross validation and param selection iterations
    classifiers = run_GridSearchCV_fit(gd_clf, X_train, y_train, X_test, y_test, label, classifiers, MODELS_OUTPUT_DIR, pickle)

    return classifiers

def run_GridSearchCV_fit(gd_clf, X_train, y_train, X_test, y_test, label, classifiers, MODELS_OUTPUT_DIR, pickle):
    logger = logging.getLogger('model-learner.model_building.run_GridSearchCV_fit')

    # fit the model
    print('start build model.')
    model = gd_clf.fit(X_train, y_train)
    print "returnValue:",gd_clf.best_score_

    print('save model.')
    if pickle:
        #name_list = joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl)
        name_list =  joblib.dump(model,label)
        #print name_list

    print('\n ==> model: ')
    #print model
    #print('\n ==> GridSearchCV: ')
    #print gd_clf
    #print('\n ==> grid scores: ')
    #print gd_clf.grid_scores_
    #print("\n ==> best ks score: %0.3f" % gd_clf.best_score_)
    #print('\n ==> best estimator: ')
    #print gd_clf.best_estimator_
    #print('\n ==> best param: ')
    #print gd_clf.best_params_
    best_parameters = dict();
    best_parameters = gd_clf.best_estimator_.get_params()
    #print best_parameters

    # plot gains chart on testing dataset
    #fig = plt.figure(1)
    plot_data=plot_gains_chart(model, X_test, y_test, label)
    #plot_img_name_prefix = os.path.split(label)
    #plot_img_name = plot_img_name_prefix[0] + "/model_plot.jpg"
    #print plot_img_name
    #plt.savefig(plot_img_name)

    return classifiers

def score_normalization_batch(proba_b):
    scores_fixed = []
    for prob in proba_b:
        score = 300 + (600 * prob)
        scores_fixed.append(score)
    return scores_fixed

def score_normalization(min, max, p):
    return min + (max * p)

def make_figure_score(proba_b, y_test, label):
    # load score base file
    name_prefix = os.path.split(label)
    print name_prefix
    f = open(settings.INPUT_DIR + 'scores_base.pkl', 'rb')
    scores_base = pickle.load(f)
    print len(scores_base)
    f.close()

    # score normalization
    scores_unfixed = score_normalization_batch(proba_b)
    scores_fixed = []
    for score in scores_unfixed:
        score_fixed = score_normalization(300, 900, scipy.stats.percentileofscore(scores_base, score, kind='mean') / 100.0)
        scores_fixed.append(score_fixed)

    black_num = sum(y_test)
    ind = argsort(scores_fixed)
    scores_new = [scores_fixed[i] for i in ind]
    labels_sorted = [y_test[i] for i in ind]
    plot_data = [sum(labels_sorted[:i])*100.0/sum(labels_sorted) for i in range(len(scores_new))]
    plt.plot(scores_new, plot_data)
    #plt.plot(range(300, 900 + step2, step2), plot_data)
    plt.plot(range(300, 900 + 12, 12), [i for i in range(0, 100+2, 2)], '.')
    #plt.axis([0, 100, 0, 100])
    plt.xlabel(u"评分", fontproperties=myfont)
    plt.ylabel(u"占违约人群百分比(%)", fontproperties=myfont)
    #plt.show()
    plot_img_name_prefix = os.path.split(label)
    plot_img_name = plot_img_name_prefix[0] + "/score_plot.png"
    print plot_img_name
    plt.savefig(plot_img_name)

def plot_gains_chart(model, X_test, y_test, label):
    y_proba = model.predict_proba(X_test)
    proba_w = [x[0] for x in y_proba]
    proba_b = [x[1] for x in y_proba]

    #dump to file
    plot_img_name_prefix = os.path.split(label)
    proba_file_name = plot_img_name_prefix[0] + "/proba_b.pkl"
    test_file_name = plot_img_name_prefix[0] + "/y_test.pkl"
    out_proba = open(proba_file_name, 'wb')
    out_test = open(test_file_name, 'wb')
    pickle.dump(proba_b, out_proba)
    pickle.dump(y_test, out_test)

    # add normalization score chart
    #make_figure_score(proba_b, y_test, label)
    #make_gains_chart(proba_b, y_test, 1, label)

def make_gains_chart(proba_b, y_test, step, label):
    # ind = argsort(y_proba)[::-1]
    ind = argsort(proba_b)
    plot_data = [0]
    for i in range(step, 100 + step, step):
        s = 0
        for i in range(len(y_test)*i / 100):
            s += y_test[ind[i]]
        plot_data.append(s * 100.0 / sum(y_test))

    plt.plot(range(0, 100+step, step), plot_data)
    plt.plot(range(0, 100+step, step), [i for i in range(0, 100+step, step)], '.')
    plt.axis([0, 100, 0, 100])
    plt.xlabel(u"占总人群百分比(%)", fontproperties=myfont)
    plt.ylabel(u"占违约人群百分比(%)", fontproperties=myfont)
    ##plt.show()
    plot_img_name_prefix = os.path.split(label)
    plot_img_name = plot_img_name_prefix[0] + "/ks_plot.png"
    print plot_img_name
    plt.savefig(plot_img_name)

    return plot_data

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
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl')

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
