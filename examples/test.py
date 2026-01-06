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
from fgqsar_gen.generator import attach_fragments
from fgqsar_gen.featurize import maccs_fp_to_array

with open("../../../test/rf_maccs_oScaffold.pkl",'rb') as f:
    model = pickle.load(f)

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

candidates,smis = attach_fragments(scaffold_core, fgs,n=200)
print(f"Generated {len(candidates)} molecules")
X = []
valid_smis = []
for smi in candidates:
    fp = maccs_fp_to_array(smi)
    if fp is not None:
        X.append(fp)
        valid_smis.append(smi)

X = np.array(X)
preds = model.predict(X)
print(preds)
print(smis)
