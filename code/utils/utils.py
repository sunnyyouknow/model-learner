# Python script to provide some additional useful functions
# this is primarily the logging script
import logging
import datetime
import os
import pickle
import settings


# define function to set up the logging
def setLog(logging_file,logtype='Master'):
    #Firstly make backup of previous run log file (Only does this for last run only).
    if os.path.exists(logging_file + '.previous'): #checks if a previous logfile exists.
        os.remove(logging_file + '.previous')
    if os.path.exists(logging_file):
        os.rename(logging_file, logging_file + '.previous')
    # set results log file (basically .info in log)
    if logtype=='Master':
        logger = logging.getLogger('Master')
    else:
        logger = logging.getLogger('Exec')
    logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler(logging_file)
    # could add log format here if required.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)

# simple function to add the input information to the log with a timestamp
def logInfoTime(logger, text):
    logger.info(text + ' %s', datetime.datetime.now().time().isoformat())


# Define a function to parse the decision tree dot file to make
# it more readable
def parseDecTree(dot_file,var_list):
    # import the regular expression library
    import re
    import os

    # open the dot file for reading
    inFile=open(dot_file,'r')
    # open the output file
    outFile = open(os.path.splitext(dot_file)[0]+'_labels.dot','w')

    # make a dictionary from the input var list
    d={}
    counter=0
    for var in var_list:
        d['X['+str(counter)+']'] = var
        counter+=1

    # use the dictionary to make a regular expression pattern for replacement
    # the key has '[' and ']' so need to escape these
    pattern = re.compile('|'.join(re.escape(key) for key in d.keys()))

    # make an output list for the file, loop over the lines in the file and
    # apply the replacement
    out=[]
    var_list=[]
    for line in inFile:
        out.append(pattern.sub(lambda x: d[x.group()],line))
        # also want a list of which variables are in the tree
        for key,var in d.items():
            if key in line:
                var_list.append(var)

    # now write out the file
    for item in out:
        outFile.write(item)

    outFile.close()
    inFile.close()

    return var_list
