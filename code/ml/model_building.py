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

def modelsBuild(np_data, y, logger):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build')

    np_data_fs = np_data
        
    logger.info('--------------------------------- Modelling - Create Test and Training Sets -----------------------------------')
    
    X_train,y_train,X_test,y_test = modfuncs.kfoldTrainingSet(np_data_fs,y,n_folds=3)
    
    # pickle the test dataset
    joblib.dump({'X_test':X_test,'y_test':y_test},settings.MODELS_OUTPUT_DIR + '/test_data.pkl')  
    
    logger.info('--------------------------------- Modelling - Basic Model Building -----------------------------------')
    classifiers = {} 
 
    # Decision Tree
    # Dictionary of the sci-kit learn logistic regression parameters
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
        classfiers = modfuncs.decTree(X_train,y_train,X_test,y_test, classifiers,'Decision Tree',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR,pickle=True, **sklearn_params)
    
    # Logistic Regression
    # Dictionary of the sci-kit learn logistic regression parameters
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
        classfiers = modfuncs.logReg(X_train,y_train,X_test,y_test, classifiers,'Logistic Regression',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR,pickle=True, **sklearn_params)
    
    # Random Forest
    # Dictionary of the sci-kit learn logistic regression parameters
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
        classfiers = modfuncs.randomForest(X_train,y_train,X_test,y_test, classifiers,'Random Forest',settings.RESULTS_OUTPUT_DIR,settings.MODELS_OUTPUT_DIR,pickle=True, **sklearn_params)  
           
    # nonProbClassifiers={} # empty dictionary to store the results
    # logger.info('--------------------------------- Plot ROC Curves -----------------------------------')
    # modfuncs.plotRocCurve(classifiers, settings.RESULTS_OUTPUT_DIR, nonProbClassifiers)
    # print 'plot ROC curves...'

if __name__ == "__main__":
	print ('Please run this script from the machine_learning_master script')
