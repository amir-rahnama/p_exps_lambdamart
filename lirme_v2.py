import re
import shap
from lime import lime_tabular
import lightgbm
import pandas as pd
import numpy as np
import sys
#sys.path.append('.')
#from data.get_data import get_data
import pickle
from scipy.stats import truncnorm, norm
import time
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

import numpy as np
from scipy.stats import truncnorm
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from utils import get_bxi, compute_mu_sigma_tilde, get_training_data_stats
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from scipy.special import expit

def custom_loss_function(y_true, y_pred, *args, **kwargs):
    total = len(y_true)
    loss_val = 0
    
    for s in range(1, total - 1):
        for i in range(total):
            if  y_true[i] < y_true[s]:
                loss_val += expit(y_true[i] < y_true[s])
    
    return loss_val

class LIRME:
    def __init__(self, data):
        self.data = data
        self.kernel_width = np.sqrt(data.shape[1]) * .75
        self.training_data = data
        self.feature_names = np.arange(self.training_data.shape[1])
        self.categorical_features = list(range(self.training_data.shape[1]))
        self.to_discretize = self.categorical_features
        self.stats()
        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(data)
        self.random_state = 10
        self.cov = np.cov(data.T)

    def kernel(self, d):
        return np.exp(-(d ** 2) / ( self.kernel_width**2))
    
    def get_bins(self, data):
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], [25, 50, 75]))
            bins.append(qts)
        return bins

    def discretize(self, data):
        ret = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                ret[feature] = int(self.lambdas[feature](ret[feature]))
            else:
                ret[:, feature] = self.lambdas[feature](
                    ret[:, feature]).astype(int)
        return ret

    def get_undiscretize_values(self, feature, values):
        mins = np.array(self.mins_all[feature])[values]
        maxs = np.array(self.maxs_all[feature])[values]

        means = np.array(self.means_all[feature])[values]
        stds = np.array(self.stds_all[feature])[values]

        minz = (mins - means) / stds
        maxz = (maxs - means) / stds
        min_max_unequal = (minz != maxz)

        ret = minz
        ret[np.where(min_max_unequal)] = truncnorm.rvs(
            minz[min_max_unequal],
            maxz[min_max_unequal],
            loc=means[min_max_unequal],
            scale=stds[min_max_unequal],
            random_state=self.random_state
        )
        return ret

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means_all:
            if len(data.shape) == 1:
                ret[feature] = self.get_undiscretize_values(
                    feature, ret[feature].astype(int).reshape(-1, 1)
                )
            else:
                ret[:, feature] = self.get_undiscretize_values(
                    feature, ret[:, feature].astype(int)
                )
        return ret
    
    def quantile_sampling(self, instance, num_samples=1000, sampling_type='quantile'):         
        # TODO: Add this for RankLIME
        
        '''if sampling_type == 'rank_lime':
            #data = np.random.normal(0, 1, num_samples* num_cols).reshape(num_samples, num_cols)
            data = np.random.multivariate_normal(np.zeros(instance.shape[0]), self.cov, num_samples)

            data = np.array(data)
            data = data * self.scaler.scale_ + instance'''
        
        #data_row = training_data[0]
        data_row = instance
        #num_samples = 1000
        num_cols = self.training_data.shape[1]
        data = np.zeros((num_samples, num_cols))
        num_cols = data.shape[1]
        first_row = self.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        
        for column in self.categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = np.random.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        
        inverse[1:] = self.undiscretize(inverse[1:])
        inverse[0] = data_row
        
        return data, inverse
        
        
    def stats(self):
        self.lambdas = {}
        self.names = {}
        self.mins_all = {}
        self.maxs_all = {}
        self.means_all = {}
        self.stds_all = {}
        self.feature_values = {}
        self.feature_frequencies = {}
        
        bins = self.get_bins(self.training_data)
        bins = [np.unique(x) for x in bins]

        for feature, qts in zip(self.to_discretize, bins):
            n_bins = qts.shape[0]  # Actually number of borders (= #bins-1)
            boundaries = np.min(self.training_data[:, feature]), np.max(self.training_data[:, feature])
            name = self.feature_names[feature]

            self.names[feature] = ['%s <= %.2f' % (name, qts[0])]
            for i in range(n_bins - 1):
                self.names[feature].append('%.2f < %s <= %.2f' %
                                           (qts[i], name, qts[i + 1]))
            self.names[feature].append('%s > %.2f' % (name, qts[n_bins - 1]))

            self.lambdas[feature] = lambda x, qts=qts: np.searchsorted(qts, x)
            discretized = self.lambdas[feature](self.training_data[:, feature])

            self.means_all[feature] = []
            self.stds_all[feature] = []
            for x in range(n_bins + 1):
                selection = self.training_data[discretized == x, feature]
                mean = 0 if len(selection) == 0 else np.mean(selection)
                self.means_all[feature].append(mean)
                std = 0 if len(selection) == 0 else np.std(selection)
                std += 0.00000000001
                self.stds_all[feature].append(std)
            self.mins_all[feature] = [boundaries[0]] + qts.tolist()
            self.maxs_all[feature] = qts.tolist() + [boundaries[1]]
        
        discretized_training_data = self.discretize(self.training_data)
        
        for feature in self.categorical_features:
            column = discretized_training_data[:, feature]

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                             float(sum(frequencies)))
            
    
    def get_exp_vals(self, model_type, samples, labels, sample_weights):
        if model_type == 'svm':    
            #scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
            #samples = scaler.fit_transform(samples)
            svr = SVR(kernel='linear')
            #print('labels', labels)
            #print('sample_weights', sample_weights)
            svr.fit(samples, labels, sample_weight=sample_weights)
            exp = svr.coef_.flatten()
            
        elif model_type == 'ridge':   
            #ridge = make_pipeline(StandardScaler(with_mean=False), Ridge())
            #kwargs = {s[0] + '__sample_weight': sample_weights for s in ridge.steps}
            ridge = Ridge(alpha=1.0)
            #print(samples.shape, labels.shape)
            ridge.fit(samples, labels, sample_weight=sample_weights)
            #exp = ridge.named_steps['ridge'].coef_
            exp = ridge.coef_
            
        return exp
        

    def get_exp_labels(self, label_type, preds, original_preds, instance_idx, top_rank_k=5):
        top_k = len(original_preds) - 1 if len(original_preds) < top_rank_k else top_rank_k
        #print('len(original_preds)', len(original_preds))
        epsilon = 0.00002
        
        if label_type == 'regression':
            labels = preds            
        elif label_type == 'score': 
            max_pred = np.max(original_preds)
            #print('max_pred', max_pred)
            labels = 1 - ((max_pred -  preds) / (max_pred + epsilon))
            #print('components', max_pred -  preds, max_pred + epsilon)
        elif label_type == 'top_k_binary':
            old_rank = np.argsort(original_preds)[::-1]
            idx_rank = np.argwhere(old_rank == top_k).flatten()[0]
            ref_pred = original_preds[idx_rank]
            labels = np.zeros(len(preds))
            label_idx = np.argwhere(preds > ref_pred).flatten()
            labels[label_idx] = 1
        
        elif label_type == 'top_k_rank':
            '''old_rank = np.argsort(original_preds)[::-1]
            idx_ref_rank = np.argwhere(old_rank == top_k).flatten()
            ref_rank = old_rank[idx_ref_rank]
            labels = []
            
            for p in preds: 
                pred_copy = original_preds.copy()
                pred_copy[instance_idx] = p
                new_rank = np.argsort(pred_copy)[::-1]
                new_rank_instance = np.argwhere(new_rank == instance_idx).flatten()
                
                if new_rank_instance < idx_ref_rank:
                    labels.append(0)
                else: 
                    labels.append(1 - (new_rank_instance/ top_k))
            labels = np.array(labels) '''
            #print(top_k)
            rank_preds = np.argsort(preds)[::-1]
            labels = np.zeros(len(preds))
            label_idx = np.argwhere(rank_preds > top_k).flatten()
            for idx in label_idx:
                labels[idx] = (1 - rank_preds[idx]) / top_k
            labels = np.array(labels)

        return labels
           
    def explain(self, instance, pred_fun, query_doc_preds, sur_type='ridge', 
                label_type='regression', instance_idx=None, top_rank=5, sample_size=1000, sampling_type='quantile'):

        data, inverse = self.quantile_sampling(instance, sample_size, sampling_type=sampling_type)
        # Changed due to bug
        scaled_data = (inverse - self.scaler.mean_) / self.scaler.scale_
        
        distances = pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric='euclidean'
        ).ravel()
        
        sample_weights = self.kernel(distances)
        x_preds = pred_fun(inverse)
        
        labels = self.get_exp_labels(label_type, x_preds, query_doc_preds, instance_idx, top_rank)
        exp = self.get_exp_vals(sur_type, scaled_data, labels, sample_weights)
        
        return exp