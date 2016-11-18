import os
dir = os.path.abspath('../')

# set the directories
INPUT_DIR = dir + '/data/'
RESULTS_OUTPUT_DIR = dir + '/results/'
MODELS_OUTPUT_DIR = dir + '/models/'
logging_file_training = dir + '/results/training.log'
logging_file_scoring = dir + '/results/scoring.log'

# dictionary listing which models to run
models = {
           'Decision Tree':False
          ,'Logistic Regression':False
          ,'Random Forest':False
          ,'Logistic Regression Grid Search':False
          ,'Decision Tree Grid Search':False
          ,'Random Forest Grid Search':False
          ,'SGD':False
          ,'Xgboost':True
         }

# feature_selection=False
# variable_scaling=False

# Set the files for the model training and scoring
train_file_name_white = 'white_lt_5000_bin.csv'
train_file_name_black = 'black_lt_1000_bin.csv'

# score_file_name = 'test_data.pkl'
score_file_name = 'xyb_yd_feature.csv'
score_file_name = 'bq_feature.csv'
model_pickle_file = 'Logistic_Regression_Grid_Search.pkl'
