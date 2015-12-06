#coding=utf-8
import numpy as np
import logging
import pprint as pp
import pickle

def pklTrainFile(INPUT_DIR, white_file_name, black_file_name, RESULTS_OUTPUT_DIR):
    logger = logging.getLogger('Master.Data_load.csvfile')
    
    #---------Load Data-----------------------
    try:
        pkl_file_white = open(INPUT_DIR + white_file_name, 'rb')
        pkl_file_black = open(INPUT_DIR + black_file_name, 'rb')
        data_white = pickle.load(pkl_file_white)
        data_black = pickle.load(pkl_file_black)

        print '----------- white data -----------'
        print(data_white.shape)
        for x in xrange(1,len(data_white)):
            print 'white %d: ' %x
            pp.pprint(data_white[x])

        print '----------- black data -----------'
        print(data_black.shape)
        for x in xrange(1,len(data_black)):
            print 'black %d: ' %x
            pp.pprint(data_black[x])

        pkl_file_white.close()
        pkl_file_black.close()

        logger.info ('Data Load White and Black Samples Completed Successfully')
        print('Data Load White and Black Samples Completed Successfully')
        
        # Merge training samples
        data_np_array =  np.concatenate((data_white, data_black))
        y_p_array = [0] * len(data_white)
        y_n_array = [1] * len(data_black)
        y_np_array = np.concatenate((y_p_array, y_n_array))
        print(data_np_array.shape)
        print(y_np_array.shape)

        return data_np_array, y_np_array

    except IOError:
        logger.info ('cannot open file %s from folder %s', file_name, INPUT_DIR)

def pklScoreFile(INPUT_DIR, score_file_name, RESULTS_OUTPUT_DIR):
    logger = logging.getLogger('Master.Data_load.csvfile')

    #---------Load Data-----------------------
    try:
        pkl_file_score = open(INPUT_DIR + score_file_name, 'rb')
        score_data = pickle.load(pkl_file_score)

        print '----------- testing data -----------'
        print(score_data.shape)
        for x in xrange(1,len(score_data)):
            print 'white %d: ' %x
            pp.pprint(score_data[x])

        pkl_file_score.close()

        logger.info ('Data Load Score Samples Completed Successfully')
        print('Data Load Score Samples Completed Successfully')

        return score_data

    except IOError:
        logger.info ('cannot open file %s from folder %s', score_file_name, INPUT_DIR)

if __name__ == "__main__":
	print ('Please run this script from the machine_learning_master script')
