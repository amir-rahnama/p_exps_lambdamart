import re
import shap
from lime import lime_tabular
import lightgbm
import pandas as pd
import numpy as np
#from ltr_exps_new import LIRME
#from lirme import LIRME
from lirme_v2 import LIRME

import sys
sys.path.append('../..')
import pickle
#from data.get_data import get_data
import time
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
from multiprocessing import Pool
from joblib import Parallel, delayed



def find_feature_idx(feature_rule):
    res = re.findall(r'feature_id_(\d+)', feature_rule)
    if len(res) > 0:
        return int(res[0])

def transform_lime_exp(exp, features):
    transform_exp = np.zeros(len(features))

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp


def find_feature_idx(feature_rule):
    res = re.findall(r'feature_id_(\d+)', feature_rule)
    if len(res) > 0:
        return int(res[0])

    
def transform_lime_exp(exp, features):
    transform_exp = np.zeros(len(features))

    for i in range(len(exp)):
        feature_idx = np.array(find_feature_idx(exp[i][0]))
        transform_exp[feature_idx] = exp[i][1]
    
    return transform_exp


def shap_exp(instances, train_data, model, sample_size):
    sample_size_background = 100
    s = shap.sample(train_data, sample_size_background)
    explainer = shap.KernelExplainer(model.predict, s)
    shap_exps = explainer.shap_values(instances, nsamples=sample_size)
    
    return shap_exps

def random_exp(size):
    return np.random.dirichlet(np.ones(size), size=1).flatten()

def lime_exp(instance, model, train_data, sample_size):
    feature_names = ['feature_id_' + str(i) for i in np.arange(train_data.shape[1])]
    lime_exp = lime_tabular.LimeTabularExplainer(train_data[:2000], 
                                             kernel_width=3, verbose=False, mode='regression',
                                             feature_names=feature_names)

    exp = lime_exp.explain_instance(instance, model.predict, num_features=train_data.shape[1],  
                                    num_samples=sample_size)
    lime_e = exp.as_list()
    lime_e_trans = transform_lime_exp(lime_e, feature_names)
    
    return lime_e_trans

def grad_exp(instance, model):
    epsilon = 0.01

    grad = np.zeros(instance.shape[0])
    
    for j in range(instance.shape[0]):
        h = np.zeros(instance.shape[0])
        h[j] = epsilon
        df_en1 = model.predict((instance + h).reshape(1, -1))[0] 
        df_en2 =  model.predict((instance - h).reshape(1, -1))[0]
        grad[j] = np.abs(df_en1 - df_en2)/ 2 * epsilon
    
    #grad = grad #/ np.sum(grad)
    
    return grad


def compute_difference(predict_fn, new_instance_copy, base_pred):
    new_pred = predict_fn(new_instance_copy)
    return np.abs(new_pred - base_pred)

def lpi_exp(instance, predict_fn, train_data):
    instance = instance.reshape(1, -1)
    importance = np.zeros(instance.shape[1])

    base_pred = predict_fn(instance)

    def compute_importance(j):
        all_feat_values = np.unique(train_data[:, j])
        across = []
        for l in range(len(all_feat_values)):
            new_instance_copy = instance.copy()
            new_instance_copy[:, j] = all_feat_values[l]
            across.append((predict_fn, new_instance_copy, base_pred))
        diffs = Parallel(n_jobs=-1)(delayed(compute_difference)(*args) for args in across)
        return np.mean(diffs)

    importance = Parallel(n_jobs=-1)(delayed(compute_importance)(j) for j in range(instance.shape[1]))

    importance_array = np.array(importance)

    return importance_array
