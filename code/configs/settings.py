import os
dir = os.path.abspath('../')

# set the values for the input data
train_file_name_white = 'data_white_yz_tag_wb'
train_file_name_black = 'data_black_yz_tag_wb'
INPUT_DIR = dir + '/data/'

# set the output directories
DATA_OUTPUT_DIR = dir + '/data/'
RESULTS_OUTPUT_DIR = dir + '/results/'
MODELS_OUTPUT_DIR = dir + '/models/'
LOGGING_FILE = dir + '/results/results_log.log'

# dictionary listing which models to include in the run and which not
models = {
          'Decision Tree':False
          ,'Logistic Regression':True
          ,'Random Forest':True
          ,'Logistic Regression Grid Search':False
          ,'Decision Tree Grid Search':False
          ,'Random Forest Grid Search':False
          ,'SGD':False
         }
         
# feature_selection=False
# variable_scaling=False

# Set the values for the model testing
input_table_exec = 'risk_scoring_ids_to_score'
input_schema_exec = 'analytics_sandpit'
output_table_exec = 'risk_scoring_output'
score_file_name = 'data_white_yz_tag_wb'
output_file = dir + '/results/output_scores.csv'
logging_file_exec = dir + '/results/execution_log.log'
model_pickle_file = MODELS_OUTPUT_DIR + 'Random_Forest.pkl'