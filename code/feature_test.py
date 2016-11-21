#!/usr/bin/python
#coding:utf-8
import sys
sys.path.append("featurizer/")

import featurizer
import test



def Process(model_path,json_string):
    #test_json = '111 | [] | [] | []'
    print "Process",json_string
    feature_string = featurizer.ProcessFeatureEx(json_string)
    print "Process",feature_string
    format_string = ",".join(str(i) for i in feature_string)
    score = test.score_one_iterm_online(model_path,format_string)
    print "returnValue:"+str(score) + "\n"
    return "returnValue:"+str(score)

def Test(model_path):
    test_json = '15995290079|[]|[{"feature":"credittotalchangtimes","value":"4"}, \
                    {"feature":"creditexpend","value":"99"},{"feature":"debitinvest","value":"99"}, \
                    {"feature":"debitonlineexpend","value":"19"},{"feature":"debitexpend","value":"27"}, \
                    {"feature":"credittotalrepayment","value":"99"},{"feature":"creditrepaymentshortwaves","value":"0"}, \
                    {"feature":"debitnum","value":"5"},{"feature":"creditnum","value":"1"},{"feature":"debitaging","value":"3.5"},\
                    {"feature":"debitenterlevel","value":"10"},{"feature":"instalmenttimes","value":"0"}, \
                    {"feature":"enterlongwaves","value":"0"},{"feature":"creditaging","value":"1.0"},\
                    {"feature":"assetlongnetgrowth","value":"-6"},{"feature":"banknum","value":"3"},\
                    {"feature":"creditexpendtimes","value":"0"},{"feature":"debitoutsum","value":"28"},\
                    {"feature":"creditonlineexpend","value":"99"},{"feature":"debitentertimes","value":"253"},\
                    {"feature":"outshortwaves","value":"-3"},{"feature":"creditrepayment","value":"14"},\
                    {"feature":"debitbalance","value":"19"},{"feature":"debitouttimes","value":"369"},\
                    {"feature":"debitentersum","value":"29"},{"feature":"debittotalchangtimes","value":"622"},\
                    {"feature":"assetshortnetgrowth","value":"-6"},{"feature":"debitshortwaves","value":"1"},\
                    {"feature":"creditshortwaves","value":"3"},{"feature":"entershortwaves","value":"-1"}]|\
                    [{"feature":"time_length","value":"3"},{"feature":"expendlevel","value":"3"}]'
    test_json = test_json + '|[{"feature":"Age","value":"3"},{"feature":"DAPPLYTERMINAL","value":"1"}]'
    score = Process(model_path,test_json)
    #print score

if __name__ == "__main__":
    if len(sys.argv) < 3:
       print "error: need 3 arguments"
    else:
       model_path = sys.argv[1]
       json_string = sys.argv[2]
       print "main:",json_string,model_path
       #Test(model_path)
       score = Process(model_path,json_string)
       #print score
