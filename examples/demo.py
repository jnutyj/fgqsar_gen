#!/usr/bin/env python3
import sys
from pathlib import Path

# add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from fgqsar_gen import design_molecules
from fgqsar_gen import scaffold
import shap
import pandas as pd
from rdkit.Chem import MACCSkeys 
import numpy as np
import pickle

###################################
# define scaffold 
###################################

### This is a simple demo, for the dataset and how to define you need to consider by yourself.

df=pd.read_csv("../models/carbonic.csv")
smiles_list=df['SMILES'].tolist()
core_smi,count=scaffold.get_main_scaffold(smiles_list)
print(core_smi)





###################################
# shap analysis to define fragment
###################################
with open("../models/X_train.pkl",'rb') as f:
    X_train=pickle.load(f)

with open("../models/rf_maccs_xScaffold.pkl",'rb') as f:
    model = pickle.load(f)

explainer=shap.TreeExplainer(model)
shap_values=explainer.shap_values(X_train)
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
top_bits = np.argsort(mean_abs_shap)[::-1][:20] # first 20 important bits
bit_to_smarts={}
for idx in top_bits:
    bit_id = idx+1
    smarts,name = MACCSkeys.smartsPatts[idx]
    bit_to_smarts[bit_id] = smarts

for i in range(len(top_bits)):
    bit=top_bits[i]
    shap_vals_bit=shap_values[:,bit]
    if np.mean(shap_vals_bit)>0:
        print(bit+1, bit_to_smarts[bit+1],np.mean(shap_vals_bit))

###################################
# generate new molecules
###################################
fgs = {
    "Me": "[*]C",
    "Cl": "[*]Cl",
    "NH2": "[*]N",
    "OMe": "[*]OC",
    "oxygen": "[*]O",
    "NH2": "[*]N",
    "CF3": "[*]C(F)(F)F"
}

scaffold_core = "c1ccc([*])cc1[*]"
#df = design_molecules(scaffold_core, fgs, "../models/rf_maccs_xScaffold.pkl", n_gen=300, topk=10)
df =  design_molecules(scaffold_core, fgs, "../../../test/rf_maccs_oScaffold.pkl", n_gen=200, topk=10)
print(df)
