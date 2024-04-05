### Data
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
import lightgbm
import pickle
from get_exp_new import lime_exp, shap_exp, random_exp, grad_exp, lpi_exp
from lirme_v2 import LIRME
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Foo')
    parser.add_argument('--d', '-dataset', dest='dataset', required=True, help='Output file name')
    args = parser.parse_args()
    
    dataset_name = args.dataset

    if not dataset_name in ['mq2008', 'web10k', 'yahoo']:
        raise Exception('Dataset must be mq2008, web10k, yahoo')
    
    #dataset_name = 'yahoo'
    print('Getting Explanations for Dataset', dataset_name)
    
    ranker = lightgbm.Booster(model_file='./models/lmart_{}_v2.txt'.format(dataset_name))
    #ranker.params['objective'] = 'binary'
    
    background_dict = pickle.load( open( "./data/{}_background_v3.p".format(dataset_name), "rb" ) )
    test = pickle.load( open( "./data/{}_test_sample_v2.p".format(dataset_name), "rb" ) )
    
    key_val = list(test.keys())[0]
    background = []
    for q in background_dict:
        background.extend(background_dict[q])
    background = np.array(background).reshape(-1, test[key_val].shape[1])
    
    lirme = LIRME(background)
    
    p_exps = {}
    pointwise_exp = ['lime', 'shap', 'lirme', 'exs_score', 'exs_top_k_binary', 'exs_top_k_rank', 'grad_d', 'random_d', 'lpi']
    
    
    e_sample_size = {
     'mq2008': 5000,  
     'web10k': 3000,
      'yahoo': 2000
    }
    
    for key in test.keys():
        print('Document explanations for query', key)
        p_exps[key] = {}
        for e in pointwise_exp:
            p_exps[key][e] = []
        q_docs = test[key]
        #pred_q_docs = ranker.predict(q_docs)
    
        for i in range(len(q_docs)): 
            print('Document explanations for instance', i)
            instance = q_docs[i]
            p_exps[key]['lime'].append(lime_exp(instance, ranker, background, sample_size=e_sample_size[dataset_name]))
            p_exps[key]['shap'].append(shap_exp(instance, background, ranker, sample_size=e_sample_size[dataset_name]))
    
            exp_lirme = lirme.explain(instance, ranker.predict, pred_q_docs, 
                        sur_type='ridge', label_type='regression', 
                        instance_idx=i, top_rank=5, sample_size=e_sample_size[dataset_name])
            p_exps[key]['lirme'].append(exp_lirme)
    
            gradient_exp = grad_exp(instance, ranker)
            p_exps[key]['grad_d'].append(gradient_exp)
    
            exp_exs_top_k_binary = lirme.explain(instance, ranker.predict, pred_q_docs,
                        sur_type='svm', label_type='top_k_binary', 
                        instance_idx=i, top_rank=5, sample_size=e_sample_size[dataset_name])
            p_exps[key]['exs_top_k_binary'].append(exp_exs_top_k_binary)
    
            exp_exs_score = lirme.explain(instance, ranker.predict, pred_q_docs, 
                        sur_type='svm', label_type='score', 
                        instance_idx=i, top_rank=5, sample_size=e_sample_size[dataset_name])
            p_exps[key]['exs_score'].append(exp_exs_score)

            exp_exs_top_k_rank = lirme.explain(instance, ranker.predict, pred_q_docs, 
                        sur_type='svm', label_type='top_k_rank', 
                        instance_idx=i, top_rank=5, sample_size=e_sample_size[dataset_name])
            p_exps[key]['exs_top_k_rank'].append(exp_exs_top_k_rank)

            lpi = lpi_exp(instance, ranker.predict, background)
            p_exps[key]['lpi'].append(lpi)
    
            #p_exps[key]['random_d'].append(random_exp(instance.shape[0]))
    
    pickle.dump(p_exps, open( "./exps/{}_pointwise_exps_v4.p".format(dataset_name), "wb" ) )
    



