import numpy as np
from scipy.stats import rankdata
import pandas as pd
import pickle
import lightgbm
import os
from lime import lime_tabular
import sys
sys.path.append('..')
from lirme_v2 import LIRME
from sklearn.utils import shuffle
from scipy.stats import spearmanr, kendalltau
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('..')
from get_exp_new import lime_exp, shap_exp, random_exp, grad_exp
from sklearn.metrics import ndcg_score, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ndcg_score, jaccard_score, mean_squared_error

def min_max(v):
    return (v - v.min()) / (v.max() - v.min())


def get_fidelity_ltr(exps, doc_values, pred_fn):
    doc_values = doc_values.copy()
    prod = np.multiply(doc_values, exps)
    output = pred_fn(doc_values.reshape(1, -1))
    output_prod = pred_fn(prod.reshape(1, -1))
    
    return mean_squared_error(output_prod, output)


def validity_completeness(exp_val, doc_values, pred_fn, background, eval_key):
    instances_explained = doc_values.copy()
    base_pred_list = pred_fn(instances_explained)
    cutoff = np.linspace(0.05, 0.5, 10)
    
    total_feat = instances_explained.shape[1]
    all_pred_diff = []
    
    for cut in cutoff:
        top_k = int(np.floor(cut * total_feat))
        #print(cut, total_feat)
    
        if eval_key == 'completeness':
            feat_selected = np.abs(exp_val).argsort()[-top_k:][::-1]
        elif eval_key == 'validity': 
            feat_selected = np.abs(exp_val).argsort()[:top_k]
        else:
            raise Exception('choose either completeness or validity')
            
        instances_explained[:, feat_selected] = np.mean(background[:, feat_selected], axis=0)
    
        new_pred_list = pred_fn(instances_explained)
        all_pred_diff.append(np.mean(np.abs(new_pred_list - base_pred_list)))
    
    return all_pred_diff


def get_infidelity(exps, doc_values, pred_fn, background, top_k_percent=0.2):
    top_k = int(top_k_percent * background.shape[1])

    doc_values_p = doc_values.copy().reshape(1, -1)

    exps_val = exps.flatten()
    top_features =  np.argsort(np.abs(exps_val))[-top_k:][::-1]
    doc_values_p[:, top_features]  =  np.mean(background[:, top_features], axis=0)

    prod = np.multiply(doc_values_p, exps_val)
    
    output_0 = pred_fn(doc_values_p)
    output_1 = pred_fn(prod)
    
    return mean_squared_error(output_0, output_1)

def get_explain_ndcg(q_exps, doc_values, pred_fn):
    doc_values_p = doc_values.copy()
    q_exp_val = q_exps.flatten()
    
    prod = np.dot(doc_values, q_exp_val)        
    #output = min_max(pred_fn(doc_values_p))
    output = pred_fn(doc_values_p)
    
    return ndcg_score([output], [prod], k=10)

def get_dpff(q_exps, doc_values, model):
    dpff_val = model.feature_importance('split')
    output = kendalltau(np.abs(q_exps), dpff_val).statistic
    return output

def get_auc(exp):
    cutoffs = np.linspace(0.05, 0.45, 10)
    auc = {}
    
    temp = np.array(exp).mean(axis=0)
    auc_ = 0
    for k in range(1, len(cutoffs) - 1):
        x = cutoffs[k] - cutoffs[k - 1]
        y = temp[k] + temp[k-1]
        auc_ += y / ( 2 * x)
    return auc_

def summarize(eval):
    eval_summary = {}
    for measure in eval.keys():
        eval_summary[measure] = {}
        for exp_name in eval[measure].keys():
            eval_summary[measure][exp_name] = {}
            if measure in ['validity', 'completeness']: 
                temp = []
                for i in range(len(eval[measure][exp_name])):
                    temp.append(get_auc(eval[measure][exp_name][i]))
                eval_summary[measure][exp_name] = np.mean(temp)
            else: 
                eval_summary[measure][exp_name] = np.nanmean(eval[measure][exp_name]) 
    return eval_summary