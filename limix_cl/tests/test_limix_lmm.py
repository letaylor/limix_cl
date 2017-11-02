# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '..')))

import limix_lmm

import numpy as np
import scipy as sp
import pandas as pd
import unittest

class BasicTestSuite(unittest.TestCase):
    """Test limix_cl functions."""
    
    def test_linear_model(self):
        # load the cars data
        f = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                            "data", "x_dat.csv")) 
        dat = pd.read_csv(f,index_col=0)
        
        # make a few more random variables
        np.random.seed(10)
        dat["rand_v1"] = np.random.randint(0,100.0,size=dat[["dist"]].shape)
        dat["rand_v2"] = np.random.normal(2*dat[["speed"]].values+2,2)
        dat["rand_v3"] = np.random.normal(2*dat[["speed"]].values+2,5)
        #dat.to_csv("cars2.csv", sep=",")
        
        # test input check
        np.testing.assert_raises(ValueError, limix_lmm.linear_model,
            y_dat=dat[["speed", "rand_v2"]].values, x_dat=dat[["rand_v1"]])
        
        # na in y data
        # dat2 = dat.copy()
        # dat2.loc[1, "speed"] = np.nan
        # np.testing.assert_warns(limix_lmm.linear_model,
        #     y_dat=dat2[["speed", "rand_v2"]].values, x_dat=dat2[["rand_v1"]])
        
        # fit model #####
        result = limix_lmm.linear_model(dat[["speed"]].values, 
            dat[["dist", "rand_v1", "rand_v2"]], 
            permute=True, n_permute=1000, permute_seed=99)
        
        # check p-values
        np.testing.assert_allclose(result["pv"].values,
            np.array([3.995168e-13, 5.716634e-01, 1.144585e-42]), 
            rtol=1e-3, atol=1e-2)
        # check beta
        np.testing.assert_allclose(result["beta"].values,
            np.array([0.165568, 0.014217, 0.486173]), 
            rtol=1e-3, atol=1e-2)
        # check ste
        np.testing.assert_allclose(result["ste"].values,
            np.array([0.022819, 0.025135, 0.035510]), 
            rtol=1e-3, atol=1e-2)
        # check x_id
        np.testing.assert_array_equal(result["x_id"].values,
            ["dist", "rand_v1", "rand_v2"])
        # check empirical p
        np.testing.assert_allclose(result["pv_empirical"].values,
            np.array([0.000999, 0.591409, 0.000999]),
            rtol=1e-3, atol=1e-2)
        
        
        # fit model with covs and permute #####
        # "dist","rand_v3" are correlated to speed, so expect rand_v2 to have
        # a smaller effect
        result = limix_lmm.linear_model(y_dat=dat[["speed"]].values, 
            x_dat=dat[["rand_v1", "rand_v2"]], 
            covariates=dat[["dist","rand_v3"]], 
            permute=True, n_permute=1000, permute_seed=8)
        
        # check p-values
        np.testing.assert_allclose(result["pv"].values,
            np.array([5.480426e-01, 3.009014e-24]), 
            rtol=1e-3, atol=1e-2)
        # check beta
        np.testing.assert_allclose(result["beta"].values,
            np.array([0.005830, 0.424948]), 
            rtol=1e-3, atol=1e-2)
        # check ste
        np.testing.assert_allclose(result["ste"].values,
            np.array([0.009705, 0.041828]), 
            rtol=1e-3, atol=1e-2)
        # check x_id
        np.testing.assert_array_equal(result["x_id"].values,
            ["rand_v1", "rand_v2"])
        # check empirical p
        np.testing.assert_allclose(result["pv_empirical"].values,
            np.array([0.597403, 0.000999]), 
            rtol=1e-3, atol=1e-2)
        
        # tests for covariance matrix
        tmp = dat[["rand_v3"]].values
        tmp -= tmp.mean(0) # subtract mean across smpls
        tmp /= tmp.std(0) # divide by std dev for feature across smpls
        sample_cov = sp.dot(tmp, tmp.T)
        
        # test for no cov matrix set for perms
        np.testing.assert_raises(ValueError, limix_lmm.linear_model,
            y_dat=dat[["speed"]].values, x_dat=dat[["rand_v1", "rand_v2"]], 
            covariance_matrix=sample_cov, permute=True)
        
        # this should give basically identical results to previous result
        # as we just made one covariate a cov mtx
        result = limix_lmm.linear_model(y_dat=dat[["speed"]].values, 
            x_dat=dat[["rand_v1", "rand_v2"]], covariates=dat[["dist"]], 
            covariance_matrix=sample_cov)
            
        # check p-values
        np.testing.assert_allclose(result["pv"].values,
            np.array([5.480426e-01, 3.009014e-24]), 
            rtol=1e-3, atol=1e-2)
        # check beta
        np.testing.assert_allclose(result["beta"].values,
            np.array([0.005830, 0.424948]), 
            rtol=1e-3, atol=1e-2)
        # check ste
        np.testing.assert_allclose(result["ste"].values,
            np.array([0.009705, 0.041828]), 
            rtol=1e-3, atol=1e-2)
        # check x_id
        np.testing.assert_array_equal(result["x_id"].values,
            ["rand_v1", "rand_v2"])

if __name__ == '__main__':
    unittest.main()