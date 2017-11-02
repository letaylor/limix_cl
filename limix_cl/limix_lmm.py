# -*- coding: utf-8 -*-

import utils

# import sys
# import os
import numpy as np
import scipy as sp
import pandas as pd
import warnings

import qvalue
import limix.qtl.qtl as qtl

#sp.random.seed(10) # set global seed

def run_permutations(function, x_dat, pv_observed, n_permute=1000, 
    permute_seed=10, min_n_perm=0, n_smaller_obs_min=15, 
    empirical_p_across_all_x=False, **kwargs):
    """
    Takes permutation information and a function. Designed to enable recursive
    p-value correction for functions. 
    
    Notes
    ------
    Generally, one should only permute the dataset that is added in the alt
        hypothesis (H1). Therefore, this function permutes the labels of x_dat, 
        breaking the identifier with covariates and y.
    If min_n_perm is > 0 then permutation loop will be broken when 
        n_prm > min_n_perm and there are n_smaller_obs_min cases where the 
        permuted p-value is smaller than the observed. Basically, this cuts off
        permutations for cases without a large effect.
    
    Parameters
    ----------
    function : function
        Function to call with any **kwargs. 
        This function should return a pandas dataframe with a pv key 
        (which is a list of p-values for each x variable).
        This function must have the following named arguments. 
            (1) x_dat (2) permute [T|F] (3) permuting [T|F]
        When called, run_permutations passes 
            (1) permuted x_dat
            (2) permute=False (so the parent function does not call run_perm)
            (3) permuting=True (so extra data (e.g. beta) not stored). 
    pv_observed : np.array
        Array of observed p-values
    n_permute : int
        Number of permutation sets to perform. 
        A good value is between 1,000 and 10,000.
    permute_seed : int
        Seed to set the random number generator for permutations to ensure they
        are reproducible.
    min_n_perm : int
        Settings for a special case of breaking permutation loop.
        Min number of perm to perform for a model.
    n_smaller_obs_min : int
        Settings for a special case of breaking permutation loop.
        Min number of perm to perform for a model.
    empirical_p_across_all_x : boolean
        If true, generates an empirical p-value per y feature across *all* x
        features. If false, generates an a empirical p-value for *each* x
        feature, then the permute be called once for each x feature. If this
        were an QTL function and x were SNPs, then set to true.
        Also if true, min_n_perm and n_smaller_obs_min effect nullified.
    
    Returns
    -------
    pv_epirical : int
        Empirical p-value
    """
    # pv_null = the distribtution of p values for each permutation
    # perm by variants. You would be interested in this in order to verify 
    # the calibration of the null. You could use this entire distribution to
    # calculate the variance across permutations.
    #pv_null = np.zeros( (n_permute, cis_snps.shape[1]), dtype=np.float64 )
    
    # for diagnostic purposes, going to run repeat the permutation process
    # len(permute_seeds) times. 
    # In the end I will have len(permute_seeds) p_permutation values.
    #pv_epirical = np.zeros( len(permute_seeds) )
    #repetition = 0
    
    # set random state using the permute_seed
    rand_state = np.random.RandomState(permute_seed)
    
    # storage of our permutations
    # at a minimum, we just need the minimum p value for each permutation
    # ...if we break early, we will use pv_null_min_filt to filter the
    # actual permutations that we used
    if empirical_p_across_all_x:
        pv_null_min = np.zeros( n_permute, dtype=np.float64 )
        pv_null_min_filt = np.zeros( n_permute, dtype=bool )
    else:
        # make dataframe that n_permute rows and n x_dat features cols
        pv_null_min = np.zeros( [n_permute, x_dat.shape[1]], dtype=np.float64 )
        pv_null_min_filt = np.zeros( [n_permute, x_dat.shape[1]], dtype=bool )
        
    # Generate the null permutations
    perm_i = 0 # the code below assumes perm_i starts off at 0
    while perm_i != n_permute:
        
        # set the permutation for this dataset.
        perm_id = rand_state.permutation(x_dat.shape[0])
        #x_dat_perm = x_dat[perm_id,:].copy() # if x were numpy array
        x_dat_perm = x_dat.iloc[perm_id,:].copy() # if x were pandas df
        # NOTE: for future interaction lmm set interaction below... 
        # perm is an array of permuted indices of individuals: 
        #   perm = SP.random.perm(N)
        #lmm.setPermutation(perm_id)
        #lmm.process() # re-run the model
        
        # now execute the function using the specified kwargs
        # this function should have the following input variables:
        # (1) model info (x_dat and y_dat) 
        # (2) permute [T|F] 
        # (3) permuting [T|F] - to tell we are
        # the function should return a dictionary containing a pv key,
        # corresponding to the p-values. This is used to perform the correction
        results_df = function( 
            x_dat=x_dat_perm, 
            permute=False, 
            permuting=True, 
            **kwargs) # assume all **kwags correspond to function
        
        if results_df is None:
            warnings.warn("Received no results in permutation loop, \
                re-running function in verbose mode with current permutation")
            
            kwargs["verbose"]=True
            results_df = function(
                x_dat=x_dat_perm, 
                permute=False, 
                permuting=True, 
                **kwargs)
            kwargs["verbose"]=False
            
        else: # get min p value for this permutation, ignoring any nan
            #pv_null_min[perm_i] = np.nanmin( results_df['pv'] )
            if empirical_p_across_all_x:
                pv_null_min[perm_i] = results_df['pv'].min(skipna=True)
                pv_null_min_filt[perm_i] = True
            else:
                pv_null_min[perm_i,:] = results_df['pv'].values
                pv_null_min_filt[perm_i,:] = True
                
            
            # get all the pvalues for this permutation
            # uncomment this if we want to return all of the permuted p-values 
            #pv_null[perm_i,:] = results_df['pv']
            
            # increment our permutation so the below code chuck is true
            perm_i += 1
            
            # for debug
            #print perm_i
            #print pv_null_min[pv_null_min_filt]
            #print pv_null_min_filt.sum()
            #sys.exit()
            
            # don't run the same number of permuations
            # if the number of permutations is > 1000 and 
            # 15 instances are smaller than min
            if min_n_perm > 0 and empirical_p_across_all_x:
                if (perm_i >= min_n_perm) and ( (pv_null_min[pv_null_min_filt] <= np.nanmin(pv_observed) ).sum() >= n_smaller_obs_min ):
                    warnings.warn("Breaking permutation loop. \
                        perm_i >= %d and %d instances <= min obs p value." %
                        (min_n_perm, n_smaller_obs_min))
                break
        
    # Now that we have null permutations, calculate the empirical pv
    # we calculate the number of times our null pvalue < smallest observed 
    # pvalue and correct based on that.
    #
    # FROM randtest in ade4 R package
    # ----------------------------------------------
    # If the alternative hypothesis is "greater", a p-value is estimated as:
    # (number of random values equal to or greater than the observed one + 
    #        1)/(number of permutations + 1).
    # The null hypothesis is rejected if the p-value is less than 
    # the significance level.
    #
    # If the alternative hypothesis is "less", a p-value is estimated as:
    # (number of random values equal to or less than the observed one +
    #        1)/(number of permutations + 1).
    # Again, the null hypothesis is rejected if the p-value is less than 
    # the significance level.
    #
    # Lastly, if the alternative hypothesis is "two-sided", the estimation 
    # of the p-value is equivalent to the one used for "greater" except
    # that random and observed values are firstly centered (using the
    # average of random values) and secondly transformed to their absolute
    # values.  Note that this is only suitable for symmetric random
    # distribution.
    if empirical_p_across_all_x:
        pv_epirical = ((pv_null_min[pv_null_min_filt] <= np.nanmin(pv_observed)).sum() + 1) / (np.float64(perm_i) + 1)
    else:
        perm_pv_less_observed = np.zeros( x_dat.shape[1], dtype=np.float64 )
        perm_pv_good = np.ma.masked_array(pv_null_min, 
            mask=np.invert(pv_null_min_filt))
        for i in xrange(x_dat.shape[1]):
            perm_pv_less_observed[i] = np.sum(perm_pv_good[:,i] <= 
                pv_observed[:,i])
        pv_epirical = (perm_pv_less_observed + 1) / (np.float64(perm_i) + 1)
    
    # code for debugging
    #assert len(pv_null_min) == perm_i
    #print "null_min: ", pv_null_min, "obs_min: ", np.nanmin(pv_observed)
    #print "number_less: ", ( pv_null_min <= np.nanmin(pv_observed) ).sum()
    #print "n_permute: ", perm_i, "new_min: ", pv_perm_min
    
    return pv_epirical


def linear_model(y_dat, x_dat, covariates=None, covariance_matrix=None, 
    permute=False, n_permute=10000, permute_seed=1, permuting=False,
    verbose=True):
    """
    Fits a simple linear model.
    
    For example, below is the model where [] indicates an optional value
    H0: y = [covariates] + [cov_mtx] + noise
    H1: y = [covariates] + x + [cov_mtx] + noise
    
    Notes
    -----
    x_dat may be a matrix, in which case each column is tested separately
        (as described further in limix code).
    if x_dat = cis SNPs and cov_mtx = kinship matrix, then permutations
        may not be completely correct. cov_mtx should be permuted along with
        the cis SNPs. Currently this is not implemented.
    NaN should not be present in any dataframes.
    
    Parameters
    ----------
    y_dat : np.array
        Numpy array of n rows and m columns used for y of model.
        Rows should be samples and m is currently assumed to be 1.
    x_dat : pd.df
        Pandas data frame of n rows and m columns used for y of model.
        Rows should be samples and m must be > 1.
    covariates : pd.df
        Pandas data frame of covariates for model. 
        Intercept term automatically added (column of 1).
    covariance_matrix : pd.df
        Pandas data frame of covariance matrix for random effect.
    permute : boolean
        Indicates if run permutations or not
    n_permute : int
        Number of permutations to run. Default = 10000 permutations.
    permuting : boolean
        Flag to tell the function if we are permuting. User should not set this.
    verbose : boolean
        If true, print extra output. 
    
    Returns
    -------
    results : pd.df
        Pandas data frame of results where each row corresponds to a row from
        x_dat. 
    """
    if y_dat.shape[1] != 1:
        raise ValueError("y_dat expected to have one column (gene or feature)")
    
    # y dat can have nan
    # if np.any(np.isnan(y_dat)):
    #     warnings.warn("detected NA in y_dat. LIMIX will automatically \
    #         subset down to common samples for this y_feature.")
    
    # set the covariates
    if covariates is not None:
        covs = covariates.copy()
        covs["b"] = np.ones((y_dat.shape[0], 1)) # intercept term n samples long
        if verbose: print "covariates:\t%s" % (" ".join(covs.columns))
        covs = covs.values # change to np array
    else:
        # make a random cov for testing
        #covs = np.random.randint(0,100,size=x_dat.shape[0]) 
        # covs is just an intercept term (column of ones)
        covs = np.ones((y_dat.shape[0], 1))
    
    # run lmm - separately fit for each variable in x_dat
    lmm = qtl.qtl_test_lmm(pheno=y_dat, snps=x_dat.values, 
                            K=covariance_matrix, covs=covs)
    lmm.process() # fit the model... not required, really.
    
    #result = {} # results dictionary
    
    # p values are of shape = (1, number x columns)
    pv = lmm.getPv()
    #result['pv'] = pv.copy() # vector of p values for each x variable
    result = pd.DataFrame(data={"pv":pv.flatten()}) # flatten copies
    
    # the permutating if clause is not really needed, 
    # but avoids extra computation
    if not permuting: 
        # get nan filter for p-values in case there are nans
        na_filt = np.isnan(pv)
        if verbose and np.any(na_filt): 
            warnings.warn("'\tnan value in p-values.")
            
        # q value calculation:
        # this would be a per feature correction.
        # for instance sometimes used in QTL studies
        # m = len(np.ravel(pv))
        # result['qv'] = pv.copy()
        # result['qv'][np.invert(na_filt)] = qvalue.estimate(
        #                                     pv[np.invert(na_filt)], m=m)
        
        # get other model info
        # result['beta'] = lmm.getBetaSNP().copy()
        # result['ste'] = lmm.getBetaSNPste().copy() # TODO check
        result['beta'] = lmm.getBetaSNP().flatten()
        result['ste'] = lmm.getBetaSNPste().flatten() # TODO check
        
        # run permutations
        if permute:
            if covariance_matrix is not None:
                raise ValueError("covariance_matrix not configured for \
                    permutations.")
            
            # run the permutations with a recursive call
            # only permute the genotypes
            result['pv_empirical'] = run_permutations( 
                function=linear_model, 
                y_dat=y_dat, 
                x_dat=x_dat, 
                pv_observed=pv,
                covariates=covariates, 
                covariance_matrix=covariance_matrix,
                n_permute=n_permute, 
                permute_seed=permute_seed,
                verbose=False)
            
            # q value calculation 
            # this would be a per feature correction.
            # for instance sometimes used in QTL studies
            # m = len(np.ravel(pv))
            # result['qv_empirical'] = pv.copy()
            # result['qv_empirical'][np.invert(na_filt)] = qvalue.estimate(
            #                                     pv[np.invert(na_filt)], m=m)
    
    result["x_id"] = x_dat.columns.values
    
    return result


