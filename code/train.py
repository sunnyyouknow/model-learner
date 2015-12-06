# Script to build models on a training dataset 
import sys;
sys.path.append("loader/")
sys.path.append("preprocess/")
sys.path.append("ml/")
sys.path.append("configs/")
sys.path.append("utils/")

import logging
import data_load
import model_building   
import utils
import pprint as pp
import settings

#Setup the logger
print '%s' %settings.LOGGING_FILE
utils.setLog(settings.LOGGING_FILE)
logger=logging.getLogger('Master')

#Log start of Full process
utils.logInfoTime(logger, 'Started')

# run hbc load data script
logger.info('--------------------------------- Data Load -----------------------------------')
utils.logInfoTime(logger, 'Started Data Load')
data_np_array, y_np_array = data_load.pklTrainFile(settings.INPUT_DIR, settings.train_file_name_white, settings.train_file_name_black, settings.RESULTS_OUTPUT_DIR)
utils.logInfoTime(logger, 'Finished Data Load')

# build models
logger.info('--------------------------------- Model Building -----------------------------------')
utils.logInfoTime(logger, 'Started Model Building')
model_building.modelsBuild(data_np_array, y_np_array, logger)
utils.logInfoTime(logger, 'Finished Model Building')

utils.logInfoTime(logger, 'Finished')

print('model training complete')