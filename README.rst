=======================
LIMIX command line tool
=======================

Command tool for fitting linear models using csv input dataframes. 

* GitHub repo: https://github.com/letaylor/limix_cl
* Free software: MIT license


Features
--------

* Easy to scale jobs over many y features using the *fold* and *iteration* options. 
* Supports permutation calculations, by permuting the sample ids of the x dataframe.


Quickstart
----------

Install::
    
    pip install git+https://github.com/letaylor/limix_cl

When installing the `limix <https://github.com/limix/limix/>`_ package, you may need to manually install `liknorm <https://github.com/limix/liknorm/>`_. 


The limix_cl installation automatically includes an execution script which can be run as shown below::
    
    # get help
    limix_cl --help
    
    # get the location of test files
    dat_dir=`pip show limix_cl | grep Location | awk '{ print $2}'`
    dat_dir=${dat_dir}"/limix_cl/tests/data"
    
    # basic call
    limix_cl --y_file ${dat_dir}"/y_dat.csv" --x_file ${dat_dir}"/x_dat.csv"
    
    # add covariates and also select a subset of samples
    limix_cl --y_file ${dat_dir}"/y_dat.csv" --x_file ${dat_dir}"/x_dat.csv" --covariates_file ${dat_dir}"/cov_dat.csv" --sample_file ${dat_dir}"/sample_subset.txt"
    
    # 100 permutations with seed as 6 (for reproducibility)
    limix_cl --y_file ${dat_dir}"/y_dat.csv" --x_file ${dat_dir}"/x_dat.csv" --covariates_file ${dat_dir}"/cov_dat.csv" --permute 100 --permute_seed 6
    
    # split the y file in to 3 batches, with current execution the 1st batch
    limix_cl --y_file ${dat_dir}"/y_dat.csv" --x_file ${dat_dir}"/x_dat.csv" --fold 3 --iteration 0
    
    # now run the 2nd batch
    limix_cl --y_file ${dat_dir}"/y_dat.csv" --x_file ${dat_dir}"/x_dat.csv" --fold 3 --iteration 1
    
    # now run the 3rd batch
    limix_cl --y_file ${dat_dir}"/y_dat.csv" --x_file ${dat_dir}"/x_dat.csv" --fold 3 --iteration 2
    
    # now run the 4rd batch - raises error since there is no 4th batch
    limix_cl --y_file ${dat_dir}"/y_dat.csv" --x_file ${dat_dir}"/x_dat.csv" --fold 3 --iteration 4
    
    # list files generated
    ls swarm-*
    
    # delete all files generated
    rm -r swarm-*


Install Notes
-------------

You may find errors relating to *machine_ffi* or *liknorm*. This is due to the `liknorm <https://github.com/limix/liknorm/>`_ package required by limix. You may need to manually install the liknorm headers to locally. Below are commands that should do that (drived from https://liknorm.readthedocs.io/en/stable)::
    
    # install liknorm headers locally
    git clone https://github.com/glimix/liknorm.git
    cd liknorm
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME
    make
    
    # install liknorm pip, passing the local install of the headers
    pip install liknorm --global-option=build_ext --global-option="-I$HOME/include/" --global-option="-L$HOME/lib"


