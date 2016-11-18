#encoding: utf-8

import json
import sys
import json
import numpy as np
import feature_setting

def GetFeatureFromJson(featureDict, jsonString):
    #print featureDict
    featureList = json.loads(jsonString)

    featureName = list()
    featureValue = list()

    flist = [0.0] * len(featureDict)
    # print feature

    for i in featureList:
        # print i
        fn = str(i["feature"])
        fv = str(i["value"])

        # print f,fv
        if not featureDict.has_key(fn):
            continue

        id = featureDict[fn]
        if(len(fv) > 0):
            v = float(fv)
            # print i["feature"], v
            flist[id] = v

    return flist

def  FeatureDictInit(feature):
    featureDict = {}
    index = 0
    for i in feature:
        featureDict[i]= index
        index = index + 1
    return featureDict

def ProcessFeatureEx(json_string):
    return ProcessFeature(feature_setting.BankFeatureTitle,
                   feature_setting.YanzhenFeatureTitle, json_string)


def ProcessFeature(BankFeatureTitle,YanzhenFeatureTitle,json_string):
    bankFeatureDict = FeatureDictInit(BankFeatureTitle)
    bank_feature = GetFeatureFromJson(bankFeatureDict,json_string.split('|')[2])
    yanzhenDict = FeatureDictInit(YanzhenFeatureTitle)
    yanzhen_feature = GetFeatureFromJson(yanzhenDict,json_string.split('|')[3])
    bank_feature.extend(yanzhen_feature)
    return bank_feature

if __name__ == '__main__':
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
    #test_json = '111 | [] | [] | []'
    feature = ProcessFeature(feature_setting.BankFeatureTitle,
                             feature_setting.YanzhenFeatureTitle, test_json)
    print feature,len(feature)
