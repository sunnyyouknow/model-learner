# -*- coding: utf-8 -*-
import sys;
sys.path.append("configs/")
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



def score_normalization_batch(proba_b):
    scores_fixed = []
    for prob in proba_b:
        score = 300 + (600 * prob)
        scores_fixed.append(score)
    return scores_fixed

def score_normalization(min, max, p):
    return min + (max * p)

def make_figure_score(proba_b, y_test, path):
    # load score base file
    name_prefix = os.path.split(path)
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
    plt.xlabel(u"评分")#, fontproperties=myfont)
    plt.ylabel(u"占违约人群百分比(%)")#, fontproperties=myfont)
    #plt.show()
    plot_img_name = pkl_saved_path + "score_plot.png"
    print plot_img_name
    plt.savefig(plot_img_name)

def make_figure_ks(proba_b, y_test, step, path):
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
    plt.xlabel(u"占总人群百分比(%)")#, fontproperties=myfont)
    plt.ylabel(u"占违约人群百分比(%)")#, fontproperties=myfont)
    ##plt.show()
    plot_img_name = path + "ks_plot.png"
    print plot_img_name
    plt.savefig(plot_img_name)



################
y_test = []
proba_b = []
operator_nane = ['yd', 'lt', 'dx']
pkl_saved_path = sys.argv[1]

# load proba_b and y_test from pkl files
for i in range(len(operator_nane)):
	prefix_path = pkl_saved_path + operator_nane[i]
	proba_file_name = prefix_path + "/proba_b.pkl"
	test_file_name = prefix_path + "/y_test.pkl"

	if os.path.exists(proba_file_name):
		proba_pkl_file = open(proba_file_name, 'rb')
		proba_data = pickle.load(proba_pkl_file)
		proba_b.extend(proba_data)

	if os.path.exists(test_file_name):
		test_pkl_file = open(test_file_name, 'rb')
		test_data = pickle.load(test_pkl_file)
		y_test.extend(test_data)

if len(proba_b) ==  len(y_test):
	# plot score chart
	make_figure_score(proba_b, y_test, pkl_saved_path)
	# plot ks chart
	make_figure_ks(proba_b, y_test, 1, pkl_saved_path)




