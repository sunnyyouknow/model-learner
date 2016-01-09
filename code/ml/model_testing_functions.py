# Script to define a bunch of useful functions for the model_execution script
import sys;
sys.path.append("../loader/")
sys.path.append("../preprocess/")
sys.path.append("../ml/")
sys.path.append("../utils/")

from sklearn.externals import joblib
import utils
import logging
from sklearn.metrics import roc_curve, auc
import model_building_functions as modFuncs
import pprint as pp

def applyModel(model_file, data, RESULTS_OUTPUT_DIR, test_pickle_file,enable_debug_print=0):
    logger=logging.getLogger('model-learner.model_scoring')

    # load model
    clf = joblib.load(model_file)

    # print clf
    # print clf.grid_scores_
    # print("Best score: %0.3f" % clf.best_score_)
    # print clf.best_params_
    # print clf.best_estimator_

    # run the model on the testing dataset
    output = clf.predict_proba(data)

    sum = 0
    #print (output.shape)
    for x in xrange(1,len(output)):
        if enable_debug_print:
            print 'prob %d: ' %x
            pp.pprint(output[x])
        if output[x][0] < 0.5:
            sum = sum + 1
    if enable_debug_print:
        print ('error rata: %d / %d') %(sum, len(data))

    return output, clf
