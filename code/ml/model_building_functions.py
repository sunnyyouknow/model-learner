from sklearn import linear_model as sklm
from sklearn import tree as sktree
from sklearn import cross_validation as cv
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing as prepro
from sklearn.externals import joblib
from sklearn import ensemble as en
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import logging
import datetime
import os
import pprint as pp


def basicTrainTest(np_data, y):
    logger = logging.getLogger('Master.Models_Buiild.basicTrainTest')

    X_train, X_test, y_train, y_test = cv.train_test_split(np_data, y, test_size=0.30, random_state=33)

    return X_train, X_test, y_train, y_test

def kfoldTrainingSet(np_data_fs,y,n_folds=10):
    # Use stratified k fold sampling to create training and test set.
    skf = cv.StratifiedKFold(y, n_folds=n_folds)
    for train_index, test_index in skf:
        X_train, X_test = np_data_fs[train_index], np_data_fs[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    return X_train,y_train,X_test,y_test
    
def logReg(X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,pickle=False, **kwargs):
    logger = logging.getLogger('Master.Models_Build.Functions.logReg')
    
    # build the classifier
    model,classifiers = calcClfModel(sklm.LogisticRegression(**kwargs),X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)  
    
    if pickle:
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl')
    
    logger.info('Started Log Reg 5 model build %s', datetime.datetime.now().time().isoformat())
    # logger.info('Optimum Log Reg Parameters: %s', pp.pformat(modelLogReg5.coef_))
    # #get the model metrics for the test data

    logger.info('Logistic Regression Model Results --------------------------')
    logger.info('Model coefficients: %s', pp.pformat(model.coef_))
    logger.info('Model intercept: %s', pp.pformat(model.intercept_))

    return classifiers

    
def randomForest(X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,pickle=False, **kwargs):
    # set logging
    logger = logging.getLogger('Master.Models_Build.Functions.randomForest')
    
    # build the classifier
    model,classifiers = calcClfModel(en.RandomForestClassifier(**kwargs),X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)  
    
    if pickle:
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl')

    logger.info('Random Forest Model Results --------------------------------')

    return classifiers
    
def decTree(X_train,y_train,X_test,y_test, classifiers,label,RESULTS_OUTPUT_DIR,MODELS_OUTPUT_DIR,pickle=False, **kwargs):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.decTree')

    # build the classifier
    model,classifiers = calcClfModel(sktree.DecisionTreeClassifier(**kwargs),X_train,y_train,X_test,y_test,label,classifiers,MODELS_OUTPUT_DIR, pickle=pickle)    
    
    # plot the decision tree
    plotDecTree(model,RESULTS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.dot')
    logger.info('Decision tree details: %s', classifiers[label])
    
    logger.info('Decision Tree Model Results --------------------------------')
    logger.info('Feature importances: %s', pp.pformat(model.feature_importances_))

    return classifiers

def calcClfModel(clf,X_train,y_train,X_test,y_test, label,classifiers,MODELS_OUTPUT_DIR, pickle=False):
    # train the model
    model = clf.fit(X_train,y_train)
    # calculate the model metrics
    metrics = modelMetrics(model,X_test,y_test,X_train,y_train,label)
    
    classifiers[label] = metrics
    
    if pickle:
        # Pickle the model and the test data (to calc metrics later on)
        joblib.dump(model,MODELS_OUTPUT_DIR + '/' + label.replace(' ','_') + '.pkl')
    
    return model,classifiers

def calcModelMetrics(model,X,y):
    # apply the model to the input test data
    test_probas_ = model.predict_proba(X)

    # Compute ROC curve and area under the curve
    test_fpr, test_tpr, test_thresholds = roc_curve(y,test_probas_[:,1])
    test_roc_auc = auc(test_fpr,test_tpr)
    return test_fpr,test_tpr,test_roc_auc

# Define a function to calculate the testing metrics associated with 
# the test (as defined by the input) data.
# the input model is an sklearn model.fit() object that has been trainied
# X_test, y_test are the test vector and class to use
# the label is a label to use to refer to the model
# and the function returns a dictionary  object with the details of the results
def modelMetrics(model, X_test, y_test, X_train,y_train, label):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('Master.Models_Build.Functions.modelMetrics')
    
    # Apply the model to the test data
    test_fpr,test_tpr,test_roc_auc = calcModelMetrics(model,X_test,y_test)
    
    # And to the training data
    train_fpr,train_tpr,train_roc_auc = calcModelMetrics(model,X_train,y_train)
    
    print("Area under ROC curve for model "+label+" test, train: %f" % test_roc_auc,train_roc_auc)
    
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

# Define function to Plot ROC curve
def plotRocCurve(classifiers, RESULTS_OUTPUT_DIR,nonProbClassifiers={}):
    #specify details of logging (So logs outputs exactly which function fed info into log)
    logger = logging.getLogger('resulst.Models_Build.plotRocCurve')

    pl.clf()
    #pl.plot(log_fpr, log_tpr, label='Logistic Regression ROC curve (area = %0.2f)' % log_roc_auc)

    pl.figure(1)
    pl.subplot(121)
    
    for key,value in classifiers.items():
        pl.plot(value['test_false_positive'], value['test_true_positive'], label=str(value['label']) + ' (area = %0.2f)' % value['test_ROC_auc'])
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Testing set ROC Curve')
    pl.legend(loc="lower right",fontsize='x-small')
    
    # if we have additional fpr/tpr points to plot lets plot them
    if len(nonProbClassifiers)>0:
        for key,cl in nonProbClassifiers.items():
            pl.scatter(cl['fpr_test'],cl['tpr_test'],marker='o')
            pl.annotate(key,xy=(cl['fpr_test'],cl['tpr_test']),xytext=(-20,20),textcoords='offset points',ha='right',va='bottom',bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0'))
    
    pl.subplot(122)
    for key,value in classifiers.items():
        pl.plot(value['train_false_positive'], value['train_true_positive'], label=str(value['label']) + ' (area = %0.2f)' % value['train_ROC_auc'])
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Training set ROC curve')
    pl.legend(loc="lower right",fontsize='x-small')
          
    savepath = os.path.join(RESULTS_OUTPUT_DIR, 'ROC_Curve.png')
    pl.savefig(savepath)
    pl.show()
    pl.close()