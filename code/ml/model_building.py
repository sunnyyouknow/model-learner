import sys;
sys.path.append("../ml/")
sys.path.append("../configs/")

from sklearn import cross_validation as cv
import numpy as np
from sklearn import decomposition as dec
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import feature_selection as fs
import model_building_functions as modfuncs
import logging
import datetime
import pprint as pp
from sklearn.externals import joblib
import settings

def validInputCheck(np_data,y,cv):
    print (np_data.shape)[0],(np_data.shape)[1],y.shape[0]
    if np_data.shape[0] != y.shape[0] :
        print  "Error feature nums does not match label nums"
        return 0
    if np_data.shape[1] == 0:
        print  "Error feature dimension is 0"
        return  0
    # check if the sample num is enough for l-fold cross validation
    if(np_data.shape[0] < cv * cv):
        print "Error Cannot have number of folds %d greater than the number of samples"%(cv)
        return  0

    if len(list(set(y))) < 2:
        print "Error only one class,can not apply trainning"
        return  0

    return 1




def modelsBuild(np_data, y, model_id,logger):
    logger = logging.getLogger('model-learner.model_building')

    classifiers = {}
    status = validInputCheck(np_data,y,3)
    if status == 0:
        return ""
    #return 0
    # split samples into training set and testing set
    logger.info('==> Create Test and Training Sets.')

    X_train, X_test, y_train, y_test = modfuncs.basic_train_test_split(np_data, y)

    joblib.dump({'X_test':X_test, 'y_test':y_test}, settings.INPUT_DIR+'/test_data.pkl')

    # build models
    logger.info('==> Building Models.')

    # Grid Search Logistic Regression
    param_grid = {
                    'penalty':('l1', 'l2')
                    ,'C':[0.0001, 0.001, 0.01, 0.1, 1, 10]
                    ,'fit_intercept':('True', 'False')
                }
    if settings.models['Logistic Regression Grid Search']==True:
        print 'run logistic regression GridSearchCV...'
        classifiers = modfuncs.logistic_regression_GridSearchCV(param_grid, X_train, y_train, X_test, y_test, classifiers, \
                                model_id, settings.RESULTS_OUTPUT_DIR, settings.MODELS_OUTPUT_DIR, pickle=True)

    # Decision Tree
    sklearn_params={
                    'criterion':'entropy'
                    ,'splitter':'best'
                    ,'max_depth':100
                    ,'min_samples_split':100
                    ,'min_samples_leaf':10
                    ,'max_features':'sqrt'
                    ,'compute_importances':None
                    ,'max_leaf_nodes':None
                    }
    if settings.models['Decision Tree']==True:
        print 'run decision tree..'
        classfiers = modfuncs.decision_tree(X_train, y_train, X_test, y_test, classifiers, \
                                'Decision Tree', settings.RESULTS_OUTPUT_DIR, settings.MODELS_OUTPUT_DIR, pickle=True, **sklearn_params)

    # Logistic Regression
    sklearn_params={
                    'penalty':'l1'
                    ,'dual':False
                    ,'tol':0.0001
                    ,'C':0.10
                    #,'fit_intercept':True
                    #,'intercept_scaling':1
                    #,'class_weight':'auto'
                    #,'random_state':None
                    }
    if settings.models['Logistic Regression']==True:
        print 'run logistic regression...'
        classfiers = modfuncs.logistic_regression(X_train,y_train,X_test,y_test, classifiers, \
                                'Logistic Regression', settings.RESULTS_OUTPUT_DIR, settings.MODELS_OUTPUT_DIR, pickle=True, **sklearn_params)

    # Random Forest
    sklearn_params={
                    'n_estimators':20
                    ,'criterion':'gini'
                    ,'max_depth':100
                    ,'min_samples_split':10
                    ,'min_samples_leaf':10
                    ,'max_features':None
                    }
    if settings.models['Random Forest']==True:
        print 'run random forest...'
        classfiers = modfuncs.random_forest(X_train, y_train, X_test, y_test, classifiers, \
                                'Random Forest', settings.RESULTS_OUTPUT_DIR, settings.MODELS_OUTPUT_DIR, pickle=True, **sklearn_params)

    if settings.models['Xgboost']==True:
        print "run Xgboost ..."
        classfiers = modfuncs.xgboost(X_train, y_train, X_test, y_test,classifiers,'xgboost',model_id,pickle=True,**sklearn_params)


if __name__ == "__main__":
	print ('Please run this script from the train script')
