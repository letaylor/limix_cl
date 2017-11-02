import numpy as np
import scipy as sp
from scipy import stats as st
import pandas as pd

import os
import warnings
import inspect
import gzip

def parallelize_yfeatures(out_dir, function, y_dat, nfolds=1000, 
    fold_i=0, debug=False, verbose=True, **kwargs):
    """
    Executes a function that takes a y dataframe (called y_dat). 
    The function should do something and return a pandas dataframe.
    
    If fold_i == 0, then the parameters are also dumped in a pickle in out_dir.
    
    Parameters
    ----------
    out_dir : string
        Path to out directory. A "data" directory will be created within
        this dir
    function : python function
        This function will be executed over a subset of y features, 
        one at a time. The function must use a parameter, y_dat.
    y_dat : pd.df
        Y data for linear model that will be swarmed over and divided using 
        folds. rows = y_features and cols = samples
    nfolds : int
        How many runs we split the list of y_dat into.
    fold_i : int
        The current iteration / fold we are on. 
        Zero based. Valid range = 0 to fold-1.
    debug : boolean
        Override nfolds and fold_i settings to run short version for debugging
        purposes. 
    verbose : boolean
        Print more output as to what the script is doing.
    
    Returns
    -------
    results : pd.df
        Pandas dataframe of results for this iteration. 
    """
    assert fold_i <= nfolds-1 and fold_i >= 0, "Invalid iteration value (%d).\
        Valid range = 0 to <= %d." % (fold_i, nfolds-1) 
    
    # write settings only if running over first fold
    if fold_i == 0:
        frame = inspect.currentframe()
        args, __, __, values = inspect.getargvalues(frame)
        with open('swarm-settings.tsv', 'w') as out_f:
            exlcude_cols = ["function", "x_dat", "y_dat", 
                "covariates", "covariance_matrix"]
            for i in args:
                if i not in exlcude_cols:
                    out_f.write("%s\t%s\n" % (i, values[i]))
            for key, value in kwargs.iteritems():
                if key not in exlcude_cols:
                    out_f.write("%s\t%s\n" % (key, value))

    # set up output directory
    out_dir = os.path.join(out_dir, 'swarm-data') # runs dir for all out files
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # get the bin of this batch...
    # by default, we only write 500 files to a directory.
    # this helps with cluster parallelization
    if not debug:
        cluster_bin = int(np.floor(fold_i / 500))
        out_dir = os.path.join(out_dir, 'bin_%d' % (cluster_bin)) 
        if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # get proper file ending
    file_ending = "csv"
    
    # get the output file
    if debug:
        out_file = os.path.join(out_dir, './../debug.%s' % (file_ending))
    else:
        out_file = os.path.join(out_dir, 
            'df%d_%d.%s' % (nfolds, fold_i, file_ending))
    if verbose: 
        print "out_file:\t%s.gz" % (out_file)
    
    # make a filter for this iteration
    n_y_features = y_dat.index.shape[0]
    Icv = np.floor( nfolds * np.arange(n_y_features)/n_y_features)
    I = Icv==fold_i
    
    # subset for these features
    y_dat = y_dat.iloc[I,:]
    
    total_y_features=np.sum(I)
    cur_i=0
    return_dat_list = []
    for y_id in y_dat.index:
        cur_i += 1
        if verbose: 
            print "%s [%d/%d]:" % (y_id, cur_i, total_y_features)
        
        y_dat_tmp = y_dat.loc[y_id,:]
        y_dat_tmp = np.expand_dims(y_dat_tmp.values, axis=1)
        
        # execute function
        result_i = function(y_dat=y_dat_tmp, **kwargs)
        return_dat_list.append(result_i)
    
    # concatenate list of results
    return_dat = pd.concat(return_dat_list)
    
    # write iteration
    with gzip.open('%s.gz' % (out_file), 'w') as f: 
        return_dat.to_csv(f, index=False, index_label=False)
    
    # reset row numbers
    return_dat.reset_index(drop=True, inplace=True)
      
    return return_dat


def mean_impute_nan(mtx):
    """
    Mean imputes nan values for each column (i.e., fill in missing data using
    mean across rows).
    
    Parameters
    ----------
    mtx : np.ndarray
        matrix with empty instances
        If using to mean impute genotypes, rows = samples & columns = variants.
    
    Returns
    -------
    mtx : np.ndarray
        with empty cases mean imputed
    cols_missing : int
        number of imputed columns
    """
    if isinstance(mtx, np.matrix):
        raise TypeError("mean impute nan function does not work on np.matrix")
    
    # get the columns (variants) that have an nan in them
    # this is boolean matrix
    missing_mtx = np.isnan(mtx).sum(0)
    if missing_mtx.max() == mtx.shape[0]: # then column with all na
        raise ValueError("cannot impute mean when a column is all na")
    cols_missing = (missing_mtx > 0)
    
    def __impute_nan__(vector):
        vector[np.isnan(vector)] = np.nanmean(vector)
        #bn.replace(vector, np.nan, bn.nanmean(vector)) # bottleneck is fast
        return vector
    
    mtx_out = mtx.copy()
    mtx_out[:,cols_missing] = np.apply_along_axis(__impute_nan__, 
        0, mtx[:,cols_missing])
    
    #assert bn.anynan(mtx) == False, "A bug occurred in mean_impute_nan"
    
    return mtx_out, cols_missing.sum()


def get_ste(pv, beta, verbose=True):
    """
    This function calculates standard errors from p-values and betas as 
    standard errors are currently missing from LIMIX.
    
    Parameters
    ----------
    pv : np.array
        array of p-values
    beta : np.array
        array of betas
    verbose : boolean
        if true raises warnings then ste != finite
    
    Returns
    -------
    ste : np.array
        standard errors
    """
    z = np.sign(beta)*np.sqrt(st.chi2(1).isf(pv))
    ste = beta/z
    
    if verbose and not np.all(np.isfinite(ste)):
        warnings.warn("ste not all finite", get_ste)
    #     print "\tpv:\t",pv[~np.isfinite(ste)][:5]
    #     print "\tbeta:\t",beta[~np.isfinite(ste)][:5]
    #     print "\tz:\t",z[~np.isfinite(ste)][:5]
    #     print "\tste:\t",ste[~np.isfinite(ste)][:5]
    #     ste[~np.isfinite(ste)] = np.nan # set inf cases to NaN
    
    return ste
