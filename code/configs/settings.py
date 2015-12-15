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
          ,'Logistic Regression Grid Search':True
          ,'Decision Tree Grid Search':False
          ,'Random Forest Grid Search':False
          ,'SGD':False
         }
         
# feature_selection=False
# variable_scaling=False

# Set the files for the model training and scoring
train_file_name_white = 'data_white_yz_tag_wb'
train_file_name_black = 'data_black_yz_tag_wb'

# score_file_name = 'test_data.pkl'
score_file_name = train_file_name_white
model_pickle_file = MODELS_OUTPUT_DIR + 'Logistic_Regression_Grid_Search.pkl'
