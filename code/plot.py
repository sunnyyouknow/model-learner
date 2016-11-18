# -*- coding: utf-8 -*-
import sys
sys.path.append("configs/")
sys.path.append("chart")
from sklearn.metrics import roc_curve, auc, make_scorer
from numpy import argsort
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='fonts/fangsong.ttf')
import matplotlib.pyplot as plt
import pylab as pl
import logging
import datetime
import os
import pprint as pp
import settings
import pickle
import scipy
import math
import copy

import echart_view

def compute_odd(probality):
    #return score_normalization_batch(probality)
    odd = []
    for i in probality:
        t = (1-i)/(i)
        if t > 20:
            t = 19
        odd.append(t*30 + 300)
    return odd


def score_normalization_batch(proba_b):
    scores_fixed = []
    for prob in proba_b:
        score = 300 + (600 * (1-prob))
        scores_fixed.append(score)
    return scores_fixed

def score_normalization(min, max, p):
    return min + ((max-min) * p)

def make_score_odd_figure(proba_b, y_test,step,path):
    # load score base file
    ind = argsort(proba_b)
    plot_data = [0]
    plot_positive = [0]
    plot_odd = [0]
    plot_proba = [0]
    for i in range(step, 100 + step, step):
        sn = 0
        sp = 0
        index = len(y_test) * i / 100 - 1
        for i in range(len(y_test) * i / 100):
            sn += y_test[ind[i]]
            if y_test[ind[i]] == 0:
                sp = sp + 1
        plot_proba.append(proba_b[ind[index]]*100)
        plot_odd.append(proba_b[ind[index]]/(1-proba_b[ind[index]]))
        plot_positive.append(sp * 100.0 / (len(y_test) - sum(y_test) + 1))
        plot_data.append(sn * 100.0 / (sum(y_test) + 1))

    #for i in  range(len(plot_positive)-1):
    #    plot_odd.append(plot_positive[i]/(plot_data[i]+1))
    print plot_proba[19],plot_proba[29]
    #print len(plot_odd),len(plot_positive),plot_proba
    plt.plot(range(0, 100 + step, step), plot_data)
    plt.plot(range(0, 100 + step, step), plot_positive)
    plt.plot(range(0, 100 + step, step), plot_odd)
    plot_img_name=pkl_saved_path + "score_odd_plot.png"
    print plot_img_name
    plt.savefig(plot_img_name)
    plt.figure()
    plt.hist(proba_b,bins = 10)
    plt.savefig(pkl_saved_path + "score_odd_plot_2.png")
    pos = []
    neg = []
    for i in  range(len(proba_b)):
        if y_test[i]:
            pos.append(proba_b[i])
        else:
            neg.append(proba_b[i])

    odds = compute_odd(pos)
    print len(set(pos)),len(pos)
    plt.figure()
    plt.hist(odds,bins = 20,normed = 1)
    plt.savefig(pkl_saved_path + "score_odd_plot_3.png")
    #plt.show()



def make_figure_score(proba_b, y_test, path, model_id):
    # load score base file
    name_prefix = os.path.split(path)
    print name_prefix
    f = open(settings.INPUT_DIR + 'scores_base_' + model_id + '.pkl', 'rb')
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
    plot_img_name = pkl_saved_path + "score_plot.png"
    print plot_img_name
    plt.savefig(plot_img_name)

def make_figure_ks(proba_b, y_test, step, path):
    #ind = argsort(y_proba)[::-1]
    ind = argsort(proba_b)
    plot_data = [0]
    for i in range(step, 100 + step, step):
        s = 0
        for i in range(len(y_test)*i / 100):
            s += y_test[ind[i]]
        #print s
        plot_data.append(s * 100.0 /(sum(y_test)+1))

    plt.plot(range(0, 100+step, step), plot_data)
    plt.plot(range(0, 100+step, step), [i for i in range(0, 100+step, step)], '.')
    plt.axis([0, 100, 0, 100])
    plt.xlabel(u"占总人群百分比(%)", fontproperties=myfont)
    plt.ylabel(u"占违约人群百分比(%)", fontproperties=myfont)
    #plt.show()
    plot_img_name = path + "ks_plot.png"
    print plot_img_name
    plt.savefig(plot_img_name)

def make_chart(proba_b, y_test, step, path):
    plt.figure()
    print np.sort(proba_b)
    ind = argsort(-np.array(proba_b))
    plot_data = [0]
    for i in range(step, 100 + step, step):
        s = 0
        for i in range(len(y_test) * i / 100):
            if y_test[ind[i]] == 1:
                s += 1
        tmp = s * 100.0 / (sum(y_test) + 1)
        plot_data.append(round(tmp,2))

    xis_data = range(0, 100 + step, step)
    print xis_data,plot_data
    plt.plot(xis_data, plot_data)
    plt.plot(xis_data,xis_data, '.')
    plt.axis([0, 100, 0, 100])
    plt.xlabel(u"占总人群百分比(%)", fontproperties=myfont)
    plt.ylabel(u"占违约人群百分比(%)", fontproperties=myfont)
    # plt.show()

    plot_img_name = path + "chart.png"
    print plot_img_name
    plt.savefig(plot_img_name)
    echart_view.Process(xis_data,plot_data)


if __name__ == "__main__":
    y_test = []
    proba_b = []

    pkl_saved_path = sys.argv[1]
    model_id = pkl_saved_path[-2:-1]
    model_id = "1"
    # load proba_b and y_test from pkl files

    prefix_path = pkl_saved_path
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

    #print proba_b,y_test
    if len(proba_b) == len(y_test):
        # plot score chart
        make_figure_score(proba_b, y_test, pkl_saved_path, model_id)
        # plot ks chart

        #aa = compute_log_odd(proba_b, y_test)
        #make_score_odd_figure(aa, y_test, 1, pkl_saved_path)
        make_figure_ks(proba_b, y_test, 1, pkl_saved_path)
        #make_chart(proba_b, y_test, 1, pkl_saved_path)
        make_score_odd_figure(proba_b, y_test, 1, pkl_saved_path)
