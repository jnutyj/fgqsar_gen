#!/usr/bin/env python3
import pickle
import numpy as np
from .featurize import maccs_fp_to_array, generate_morgan_fingerprint


class RFQSAR:
    def __init__(self,model_path):
        with open(model_path,'rb') as f:
            self.model=pickle.load(f)
        
        
        ### TODO: currently the model is trained iwht maccs, but if model trained with morgan is also provided, then it requires to provide options.
        
    def predict_results(self,mols_list):
        X = []
        valid_mols=[]
        for mol in mols_list:
            fp = maccs_fp_to_array(mol)
            if fp is not None:
                X.append(fp)
                valid_mols.append(mol)
                    

        X = np.array(X)
        preds = self.model.predict(X)
        return valid_mols, preds

    def pick_top_hits(self,valid_mols,preds,n=10):
        """valid_mols, preds are the output from predict function, n means the top nth molecules"""
        idx = np.argsort(preds)[::-1][:n]
        top_hits = [(valid_mols[i],preds[i]) for i in idx]
        return top_hits
        
            
            
