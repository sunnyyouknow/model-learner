# Script to execute previously build models on a testing dataset
import sys;
import numpy
from scipy import stats

sys.path.append("loader/")
sys.path.append("preprocess/")
sys.path.append("ml/")
sys.path.append("configs/")
sys.path.append("utils/")

import logging
import data_load
import pickle
import data_load_csv
import datetime
import utils
import settings
import model_testing_functions as mexec

def score_csv():
    #Setup the logger
    print '%s' %settings.logging_file_scoring
    logger = utils.setLog(settings.logging_file_scoring, logtype='Exec')
    logger = logging.getLogger('model-learner.test')

    logger.info('Start testing: %s', datetime.datetime.now().time().isoformat())

    logger.info('==> Load Data.')
    data = data_load_csv.csv_score_file(settings.INPUT_DIR, settings.score_file_name)

    logger.info('==> Preprocessing data.')


    logger.info('==> Apply Data.')
    output,clf = mexec.applyModel(settings.MODELS_OUTPUT_DIR +'model_'+ settings.model_pickle_file, data, settings.RESULTS_OUTPUT_DIR, settings.MODELS_OUTPUT_DIR + 'test_data.pkl')

    logger.info('Finish testing: %s', datetime.datetime.now().time().isoformat())
    #print output[0]
    print score_normalization(300,900,output[0][0])
    #print('model testing complete')

def score_one_iterm_online(model_path,feature_string):
    #feature_string =  "1,1,1,-1,1,1";
    logger = utils.setLog(settings.logging_file_scoring, logtype='Exec')
    logger = logging.getLogger('model-learner.test')

    logger.info('Start testing: %s', datetime.datetime.now().time().isoformat())
    # transform string to numpy array
    np_data = numpy.fromstring(feature_string, dtype=int, sep=",")

    np_data = np_data.reshape(1,-1)
    '''
    missing_value_model_path = model_path + ".miss_value_model"
    if os.path.exists(missing_value_model_path)
        np_data = missing_value.load_miss_value_mode(np_data,missing_value_model_path)
    '''
    #print  np_data.shape
    model_id = model_path[-21:-20]
    output,clf = mexec.applyModel(model_path, np_data, settings.RESULTS_OUTPUT_DIR, settings.MODELS_OUTPUT_DIR + 'test_data.pkl')
    #print np_data
    score = score_normalization(300,900,output[0][0])
    score_fixed = cal_real_score(score, model_id)
    print "returnValue:",score_fixed
    logger.info('Finish testing: %s', datetime.datetime.now().time().isoformat())
    return score_fixed

def score_normalization(min_v,max_v,current_v):
    return min_v + (max_v - min_v)*current_v

def cal_real_score(score, model_id):
    f = open(settings.INPUT_DIR + "scores_base_" + model_id + ".pkl")
    scores_list = pickle.load(f)
    f.close()
    percentile = stats.percentileofscore(scores_list, score)
    return score_normalization(300, 900, percentile / 100.0)


if __name__ == "__main__":
    if len(sys.argv) < 3:
       print "error: need 3 arguments"
    else:
       model_path = sys.argv[1]
       feature_string = sys.argv[2]
       score_one_iterm_online(model_path,feature_string)
#score_csv()
#score_one_iterm_online("")
