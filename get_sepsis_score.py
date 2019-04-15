#!/usr/bin/env python3

import numpy as np
import os
import sys

import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib


def mylen(temp_list):
    temp_list = [temp_value for temp_value in temp_list if not np.isnan(temp_value)]
    return len(temp_list)


fun_list = [max, min, np.mean, np.std, 'skew', mylen]
fun_str = ['max', 'min', 'mean', 'std', 'skew', 'len']

ori_names = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
             'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
             'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
             'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
             'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
             'HospAdmTime']
column_names = []
for each_columns in ori_names:
    for each_fun in fun_str:
        column_names.append(each_columns + '_' + str(each_fun))


def get_sepsis_score(temp_model, values):
    data_test = values.drop(['name', 'SepsisLabel'], axis=1)
    temp_scores = temp_model.predict_proba(data_test)
    temp_scores = temp_scores[:, 1]
    temp_labels = (temp_scores >= 0.02)
    return temp_scores, temp_labels


def read_challenge_data(file_name):
    each_df = pd.read_csv(file_name, sep='|', encoding='utf-8')
    each_tongji = pd.DataFrame()
    for i in each_df.index:
        if i == 0:
            each_tongji = each_tongji.append([each_df.iloc[:i + 1, :-2].agg(fun_list).values.T.flatten()])
        else:
            each_tongji = each_tongji.append([each_df.iloc[:i, :-2].agg(fun_list).values.T.flatten()])
    each_tongji.columns = column_names
    each_tongji.reset_index(drop=True, inplace=True)
    each_tongji['name'] = file_name
    return pd.concat([each_df, each_tongji], axis=1)


if __name__ == '__main__':
    # read data
    model = joblib.load('xgboost.model')
    data = read_challenge_data(sys.argv[1])

    # make predictions
    if data.size != 0:
        scores, labels = get_sepsis_score(model, data)
    # write results
    with open(sys.argv[2], 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        if data.size != 0:
            for (s, l) in zip(scores, labels):
                f.write('%g|%d\n' % (s, l))
