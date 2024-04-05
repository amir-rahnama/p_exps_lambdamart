"""
The code is taken from:
https://github.com/dgarreau/tabular_lime_theory/blob/
under utils/aux_functions.py.
"""

import numpy as np
from scipy.stats import truncnorm, norm


def get_training_data_stats(train,p):
    """
    This function computes training data summaries in the same way Tabular 
    LIME does. 
    
    INPUT:
        train: training data (size (n_train,dim))
        p: number of bins along each dimension
        
    OUTPUT:
        my_stats["means"]: empirical mean on each bin
        my_stats["stds"]: standard deviation on each bin
        my_stats["mins"]: left side of the bin
        my_stats["maxs"]: right side of the bin
        my_stats["feature_values"]: bin indices
        my_stats["feature_frequencies"]: proba to choose the bin (1/p in our case)
        my_stats["bins"]: bins boundaries
        
    """
    n_train,dim = train.shape
    my_stats = {}
    my_stats["means"]               = {}
    my_stats["stds"]                = {}
    my_stats["mins"]                = {}
    my_stats["maxs"]                = {}
    my_stats["feature_values"]      = {}
    my_stats["feature_frequencies"] = {}
    my_stats["bins"] = {}
    
    for j in range(dim):
        data_along_j = train[:,j]
        box_boundaries_along_j = np.percentile(data_along_j, np.arange(p+1)*100/p)
        my_stats["means"][j] = []
        my_stats["stds"][j]  = []
        my_stats["mins"][j]  = []
        my_stats["maxs"][j]  = []
        my_stats["feature_values"][j] = np.arange(p,dtype=float)
        my_stats["feature_frequencies"][j] = list((1/p)*np.ones((p,)))
        my_stats["bins"][j] = box_boundaries_along_j[1:-1]
        
        for b in range(p):
            left  = box_boundaries_along_j[b]
            right = box_boundaries_along_j[b+1]
            
            select_bool = (left <= data_along_j) * (data_along_j <= right)
            
            selection = data_along_j[select_bool]
            
            my_stats["means"][j].append(np.mean(selection))
            my_stats["stds"][j].append(np.std(selection))
            my_stats["mins"][j].append(left)
            my_stats["maxs"][j].append(right)
    
    return my_stats

def get_bxi(xi,my_stats):
    """
    This function gets the bin indices for xi given summary statistics.
    
    INPUT
        xi: example to explain (size (dim,))
        my_stats: summary statistics (see get_training_data_stats)
        
    OUTPUT:
        bxi: bin indices (size (dim,))
    
    """
    p = len(my_stats["mins"][0])
    dim = xi.shape[0]
    
    bxi = np.zeros((dim,),dtype=int)
    for j in range(dim):
        for b in range(p):
            # b = bxi_j if xi_j belongs to the bin
            if my_stats["mins"][j][b] <= xi[j] and xi[j] <= my_stats["maxs"][j][b]:
                bxi[j] = b
                break
            
    return bxi

def compute_mu_sigma_tilde(my_stats):
        """
        This function computes the mean of the truncated Gaussians on all the 
        d-dimensional bins.

        INPUT:
            my_stats: summary statistics (see get_training_data_stats)

        OUTPUT:
            mutilde (size (dim,p))

        """

        dim = len(my_stats["means"])
        p = len(my_stats["mins"][0])

        mutilde = np.zeros((dim,p))
        sigmatilde = np.zeros((dim,p))


        for j in range(dim):
            for b in range(p):
                mu    = my_stats["means"][j][b]
                sigma = my_stats["stds"][j][b]
                left  = (my_stats["mins"][j][b] - mu) / sigma
                right = (my_stats["maxs"][j][b] - mu) / sigma

                #mutilde[j,b] = mu + (norm.pdf(left) - norm.pdf(right)) / (norm.cdf(right) - norm.cdf(left))
                mutilde[j, b] = mu - sigma * (norm.pdf(right) - norm.pdf(left)) /  (norm.cdf(right) - norm.cdf(left))
                _first_degree = (right * norm.pdf(right) - left * norm.pdf(left)) / (norm.cdf(right) - norm.cdf(left))
                _second_degree = (norm.pdf(right) - norm.pdf(left)) / (norm.cdf(right) - norm.cdf(left))

                sigmatilde[j, b] = sigma**2 * (1 -  _first_degree - _second_degree**2)

        return mutilde, sigmatilde
    
    
def idcg(relevance, alternate=True):
    """
    Calculate ideal discounted cumulative gain (maximum possible DCG).

    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    # guard copy before sort
    rel = np.asarray(relevance).copy()
    rel.sort()
    return dcg(rel[::-1], alternate)