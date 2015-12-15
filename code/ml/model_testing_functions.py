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

def applyModel(model_file, data, RESULTS_OUTPUT_DIR, test_pickle_file):
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

    print (output.shape)
    for x in xrange(1,len(output)):
        print 'prob %d: ' %x
        pp.pprint(output[x])

    return output, clf
