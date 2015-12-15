# Script to execute previously build models on a testing dataset 
import sys;
sys.path.append("loader/")
sys.path.append("preprocess/")
sys.path.append("ml/")
sys.path.append("configs/")
sys.path.append("utils/")

import logging
import data_load
import datetime
import utils
import settings
import model_testing_functions as mexec

#Setup the logger
print '%s' %settings.logging_file_scoring
logger = utils.setLog(settings.logging_file_scoring, logtype='Exec') 
logger = logging.getLogger('model-learner.test')
    
logger.info('Start testing: %s', datetime.datetime.now().time().isoformat())

logger.info('==> Load Data.')
data = data_load.pkl_score_file(settings.INPUT_DIR, settings.score_file_name, settings.RESULTS_OUTPUT_DIR)

logger.info('==> Preprocessing data.')


logger.info('==> Apply Data.')
output,clf = mexec.applyModel(settings.model_pickle_file, data, settings.RESULTS_OUTPUT_DIR, settings.MODELS_OUTPUT_DIR + 'test_data.pkl')

logger.info('Finish testing: %s', datetime.datetime.now().time().isoformat())

print('model testing complete')
