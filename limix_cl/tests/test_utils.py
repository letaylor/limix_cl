# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '..')))

import utils
import limix_lmm

import pandas as pd
import numpy as np
import shutil
import unittest

class BasicTestSuite(unittest.TestCase):
    """Test util functions."""
    
    def test_mean_impute_nan(self):
        # test make sure matrix input caught
        inpt = np.matrix([[1, np.nan, np.nan, 5], 
                          [3, 4, 5, np.nan], 
                          [3, 3, np.nan, -1]])
        out = np.matrix([[1, 3.5, 5, 5], 
                          [3, 4, 5, 2.0], 
                          [3, 3, 5, -1]])
        np.testing.assert_raises(TypeError, utils.mean_impute_nan, inpt)
        
        # test the impute function
        inpt2 = np.asarray(inpt)
        np.testing.assert_array_equal(utils.mean_impute_nan(inpt2)[0],
            np.asarray(out))
        assert utils.mean_impute_nan(inpt2)[1] == 3
        
        # test to make sure one col full of na is caught
        inpt2[1,2] = np.nan
        np.testing.assert_raises(ValueError, utils.mean_impute_nan, inpt2)
    
    def test_get_ste(self):
        # test ste estimation
        pv = np.array([9.99991e-10, 2.08721e-06, 0.000100015, 9.99984e-10])
        beta = np.array([0.274563, 0.744085, 0.0866211, -0.633619])
        ste = np.array([0.0435844, 0.153911, 0.0219806, 0.100581])
        # print np.isclose(utils.get_ste(pv, beta), ste, rtol=0.1)
        # print utils.get_ste(pv, beta)
        np.testing.assert_allclose(utils.get_ste(pv, beta),
            ste, rtol=1e-3, atol=1e-2)
        
        # TODO: test warning when ste not finite
        #np.testing.assert_warns()
    
    def test_parallelize_yfeatures(self):
        # load the test data
        f = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                            "data", "y_dat.csv")) 
        y_dat = pd.read_csv(f,index_col=0)
        
        f = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                            "data", "x_dat.csv")) 
        x_dat = pd.read_csv(f,index_col=0)
        
        f = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                            "data", "cov_dat.csv")) 
        cov_dat = pd.read_csv(f,index_col=0)
        
        # check invalid iteration value
        np.testing.assert_raises(AssertionError, 
            utils.parallelize_yfeatures,
            out_dir = os.getcwd(), function = limix_lmm.linear_model, 
            y_dat = y_dat.T, x_dat = x_dat, covariates = cov_dat, 
            nfolds=2, fold_i=2)
        
        # run a trial
        result = utils.parallelize_yfeatures(
            out_dir = os.getcwd(), 
            function = limix_lmm.linear_model, 
            y_dat = y_dat.T, x_dat = x_dat, covariates = cov_dat, 
            nfolds=2, fold_i=1, 
            permute=True, n_permute=100, permute_seed=2)
             
        np.testing.assert_allclose(result["pv"].values[:4],
            np.array([8.949425e-13, 8.963373e-01, 1.723188e-10, 4.810727e-01]),
            rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(result["pv_empirical"].values[:4],
            np.array([0.009901, 0.930693, 0.009901, 0.534653]),
            rtol=1e-3, atol=1e-2)
        
        # clean up output
        try:
            os.remove("swarm-settings.tsv")
        except OSError:
            pass
        shutil.rmtree("swarm-data", ignore_errors=True)

if __name__ == '__main__':
    unittest.main()