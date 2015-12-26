# Script to build models on a training dataset
import sys;
sys.path.append("loader/")
sys.path.append("preprocess/")
sys.path.append("ml/")
sys.path.append("configs/")
sys.path.append("utils/")

import logging
import data_load
import data_load_csv
import model_building
import utils
import pprint as pp
import settings


def  offline_train():
#Setup the logger
    print '%s' %settings.logging_file_training
    utils.setLog(settings.logging_file_training)
    logger=logging.getLogger('model-learner.train')

    #Log start of Full process
    utils.logInfoTime(logger, 'Started')

    # run hbc load data script
    logger.info('==> Load Data.')
    utils.logInfoTime(logger, 'Started Data Load')
    data_np_array, y_np_array = data_load_csv.csv_train_file(settings.INPUT_DIR, settings.train_file_name_white, settings.train_file_name_black)
    utils.logInfoTime(logger, 'Finished Data Load')

    # preprocessing featurizer sample data
    logger.info('==> Preprocessing feature data.')
    utils.logInfoTime(logger, 'Started Preprocessing')

    utils.logInfoTime(logger, 'Finished Preprocessing')

    # build models
    logger.info('==> Build Model.')
    utils.logInfoTime(logger, 'Started Model Building')
    model_building.modelsBuild(data_np_array, y_np_array, 'model_',logger)
    utils.logInfoTime(logger, 'Finished Model Building')

    utils.logInfoTime(logger, 'Finished')

    print('model training complete')


def online_traning_api(input_file_name,model_id):
    print '%s' %settings.logging_file_training
    utils.setLog(settings.logging_file_training)
    logger=logging.getLogger('model-learner.train')

    #Log start of Full process
    utils.logInfoTime(logger, 'Started')

    # run hbc load data script
    logger.info('==> Load Data.')
    utils.logInfoTime(logger, 'Started Data Load')
    data_np_array, y_np_array = data_load_csv.csv_train_from_one_file(input_file_name);
    utils.logInfoTime(logger, 'Finished Data Load')

    # preprocessing featurizer sample data
    logger.info('==> Preprocessing feature data.')
    utils.logInfoTime(logger, 'Started Preprocessing')

    utils.logInfoTime(logger, 'Finished Preprocessing')

    # build models
    logger.info('==> Build Model.')
    utils.logInfoTime(logger, 'Started Model Building')
    model_building.modelsBuild(data_np_array, y_np_array,model_id,logger)
    utils.logInfoTime(logger, 'Finished Model Building')

    utils.logInfoTime(logger, 'Finished')

    #print('model training complete')

#online_traning_api()
#offline_train()

if len(sys.argv) < 3:
    print "error: need 3 arguments"
else:
    train_csv_file_path = sys.argv[1]
    model_saved_path = sys.argv[2]
    online_traning_api(train_csv_file_path,model_saved_path)
