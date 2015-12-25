#coding=utf-8
import numpy as np
import logging
import pprint as pp
import pandas as pd


def  load_csv_file(file_name):
    logger = logging.getLogger('model-learner.data_load.csv_train_file')
    df = pd.read_csv(file_name, header = 0)
    #print df
    # for tab delimited use:
    # df = pd.read_csv(input_file, header = 0, delimiter = "\t")
    print df.shape
    # put the original column names in a python list
    original_headers = list(df.columns.values)

    # remove the non-numeric columns
    df = df._get_numeric_data()


    # put the numeric column names in a python list
    numeric_headers = list(df.columns.values)

    # create a numpy array with the numeric values for input into scikit-learn
    numpy_array = df.as_matrix()
    return  numpy_array

def csv_train_file(INPUT_DIR, white_file_name, black_file_name):
    logger = logging.getLogger('model-learner.data_load.pkl_train_file')

    try:
        print INPUT_DIR,white_file_name
        white_data = load_csv_file(INPUT_DIR + white_file_name);
        black_data = load_csv_file(INPUT_DIR + black_file_name);

        print 'white data:'
        print(white_data.shape)
        # for x in xrange(1,len(data_white)):
        #     print 'white %d: ' %x
        #     pp.pprint(data_white[x])

        print 'black data:'
        print(black_data.shape)
        # for x in xrange(1,len(data_black)):
        #     print 'black %d: ' %x
        #     pp.pprint(data_black[x])


        logger.info ('==> Data Load White and Black Samples Completed Successfully')
        print('Data Load White and Black Samples Completed Successfully')

        # Merge training samples
        data_np_array =  np.concatenate((white_data, black_data))
        y_p_array = [0] * len(white_data)
        y_n_array = [1] * len(black_data)
        y_np_array = np.concatenate((y_p_array, y_n_array))
        print 'X:'
        print(data_np_array.shape)
        print 'y:'
        print(y_np_array.shape)

        return data_np_array, y_np_array

    except IOError:
        logger.info ('cannot open file %s from folder %s', file_name, INPUT_DIR)

def csv_score_file(INPUT_DIR, score_file_name):
    logger = logging.getLogger('model-learner.data_load.pkl_score_file')

    try:
        score_data = load_csv_file(INPUT_DIR + score_file_name);

        # print '----------- testing data -----------'
        # print(score_data.shape)
        # for x in xrange(1,len(score_data)):
        #     print 'white %d: ' %x
        #     pp.pprint(score_data[x])

        logger.info ('==> Data Load Score Samples Completed Successfully')
        print('Data Load Score Samples Completed Successfully')

        return score_data

    except IOError:
        logger.info ('cannot open file %s from folder %s', score_file_name, INPUT_DIR)

#for debug
#csv_train_file("..//data/","white_dx_5000_bin.csv","black_dx_1000_bin.csv");
#load_csv_file("../data/black_dx_1000_bin.csv")


if __name__ == "__main__":
    print ('Please run this script from train and test script')
