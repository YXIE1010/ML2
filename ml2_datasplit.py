#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:16:02 2024

Code to split preprocessed radar observations
into training, validation, and testing datasets
which are used as model input


@author: Yan Xie (yanxieyx@umich.edu)
"""

# load modules
import numpy as np
import numpy.ma


# define filepaths
filepath = '/your/input/filepath/for/preprocessed-data/'
fpathout = '/your/output/filepath/for/model-input-data/'

flagmon = 0

for iyear in range(2015, 2017):
    if iyear == 2015:
        monlist = np.arange(3,13)
    elif iyear == 2016:
        monlist = np.arange(1,3)
        
    for imon in range(monlist[0], monlist[-1]+1):
        fname = 'data_preprocess' + '{:4d}'.format(iyear) + '{:02d}'.format(imon) +'.npy'
        with open(filepath+fname, 'rb') as fin:
            datatot = np.load(fin)
            idxtabstot = np.load(fin) 
            rag = np.load(fin)      
            idxmltot = np.load(fin)
            scalertot1 = np.load(fin)
        
        # reduce the precision
        datatot = datatot.astype(np.float32)
        idxtabstot = idxtabstot.astype(np.int32)
        idxmltot = idxmltot.astype(np.int8)
        
        # now split the prepared data
        nsec = datatot.shape[0]
        ntrain = round(nsec * 0.85)
        nvalid = round(nsec * 0.05)
        ntest = nsec - ( ntrain+nvalid ) 
        
        if flagmon == 0:
            # preparing the training data
            Datatrain = datatot[0:ntrain, :, :, :]
            tabstrain = idxtabstot[0:ntrain, :]
            idxmltrain = idxmltot[0:ntrain]
            
            # preparing the validating data
            Datavalid = datatot[ntrain:ntrain+nvalid, :, :, :]
            tabsvalid = idxtabstot[ntrain:ntrain+nvalid, :]
            idxmlvalid = idxmltot[ntrain:ntrain+nvalid]
            
            # preparing the test data
            Datatest = datatot[ntrain+nvalid:nsec, :, :, :]
            tabstest = idxtabstot[ntrain+nvalid:nsec, :]
            idxmltest = idxmltot[ntrain+nvalid:nsec]
            
            flagmon = 1
            
        elif flagmon == 1:
            # preparing the training data
            Datatrain = np.concatenate( (Datatrain, datatot[0:ntrain, :, :, :]), axis=0)
            tabstrain = np.concatenate( (tabstrain, idxtabstot[0:ntrain, :]), axis=0)
            idxmltrain = np.concatenate( (idxmltrain, idxmltot[0:ntrain]) ) 
            
            # preparing the validating data
            Datavalid = np.concatenate( (Datavalid, datatot[ntrain:ntrain+nvalid, :, :, :]), axis=0)
            tabsvalid = np.concatenate( (tabsvalid, idxtabstot[ntrain:ntrain+nvalid, :]), axis=0)
            idxmlvalid = np.concatenate( (idxmlvalid, idxmltot[ntrain:ntrain+nvalid]) )
            
            # preparing the test data
            Datatest = np.concatenate( (Datatest,datatot[ntrain+nvalid:nsec, :, :, :]), axis=0)
            tabstest = np.concatenate( (tabstest, idxtabstot[ntrain+nvalid:nsec, :]), axis=0)
            idxmltest = np.concatenate( (idxmltest, idxmltot[ntrain+nvalid:nsec]) )
            
        del datatot, idxtabstot, rag, idxmltot
        del fin  
        

##### save the split model input data #####
with open(fpathout + 'training.npy', 'wb') as fout_train:
    np.save(fout_train, Datatrain, allow_pickle=True)
    np.save(fout_train, tabstrain, allow_pickle=True)
    np.save(fout_train, idxmltrain, allow_pickle=True)

        
with open(fpathout + 'validation.npy', 'wb') as fout_valid:
    np.save(fout_valid, Datavalid, allow_pickle=True)
    np.save(fout_valid, tabsvalid, allow_pickle=True)
    np.save(fout_valid, idxmlvalid, allow_pickle=True)        
            
with open(fpathout + 'testing.npy', 'wb') as fout_test:
    np.save(fout_test, Datatest, allow_pickle=True)
    np.save(fout_test, tabstest, allow_pickle=True)
    np.save(fout_test, idxmltest, allow_pickle=True)            
    
