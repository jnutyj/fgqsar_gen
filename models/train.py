#!/usr/bin/env python3
import pandas as pd
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, AllChem, PandasTools, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pickle
import os

def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit molecule object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        # Optional: Sanitize molecule, add H, etc. as needed
        # Chem.SanitizeMol(mol)
        return mol
    except:
        return None
    
def generate_descriptors(mol,selected_descriptor_names):
    calculator=MoleculeDescriptors.MolecularDescriptorCalculator(selected_descriptor_names)
    return list(calculator.CalcDescriptors(mol))



def generate_morgan_fingerprint(mol, radius=2, nbits=2048):
    """Generate Morgan fingerprint (ECFP4 equivalent) as a numpy array."""
    if mol is None:
        return np.zeros(nbits, dtype=int)
    # Use the FingerprintGenerator for a consistent API
    #fgen = GetMorganFingerprintGenerator(radius=radius, fpSize=nbits)
    fgen = GetMorganGenerator(radius=radius, fpSize=nbits)
    # Convert bit vector to a numpy array
    arr = np.zeros((1,), dtype=np.int8)
    fp = fgen.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



def maccs_fp_to_array(mol):
    """
    Generates MACCS fingerprints as a NumPy array from a SMILES string.
    """
    #mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Generate the MACCSKeys bit vector
        fp = MACCSkeys.GenMACCSKeys(mol)
        # Convert the bit vector to a NumPy array
        arr = np.zeros((1,), dtype=np.int8) # Use int8 for binary data
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return None



def extract_X_Y(fname, selected_descriptor_names, y_col_name, smiles_col_name,fp_selection="maccs"):
    #df=fname
    df = pd.read_csv(fname, engine='python')
    smiles_list=df[smiles_col_name].tolist()
    mols=[smiles_to_mol(s) for s in smiles_list]
    valid_mols=[mol for mol in mols if mol is not None]

    if len(selected_descriptor_names)==0:
        
        descriptor_data=[]
        descriptors_df = pd.DataFrame(descriptor_data)
    else:
        descriptor_data=[generate_descriptors(mol,selected_descriptor_names) for mol in valid_mols]
        descriptors_df = pd.DataFrame(descriptor_data, columns=selected_descriptor_names)

    if fp_selection=="maccs":
        fp_data=[ maccs_fp_to_array(mol) for mol in valid_mols]
        fp_np=np.vstack(fp_data)
        #print(fp_np)
        fp_df=pd.DataFrame(fp_np, columns=[f'FP_{i}' for i in range(fp_np.shape[1])])
    elif fp_selection=="morgan":
        fp_data=[ generate_morgan_fingerprint(mol) for mol in valid_mols]
        fp_np=np.vstack(fp_data)
        #print(fp_np)
        fp_df=pd.DataFrame(fp_np, columns=[f'FP_{i}' for i in range(fp_np.shape[1])])
    elif fp_selection== None:
        fp_data=[]
        fp_np=[]
        #print(fp_np)
        fp_df=pd.DataFrame(fp_np)
        
    #print(descriptors_df)
    #print(len(fp_data[0]))
    
    #print(fp_df)
    X_df = pd.concat([descriptors_df, fp_df], axis=1)
    #print(X_df)
    X = X_df.values
    #print(X)
    Y=df[y_col_name].values
    return X,Y





if __name__ == "__main__":



    fname='carbonic.csv'
    smiles_column_name='SMILES'
    y_column_name='pIC50'
    no_descriptor_names=[]
    x0,y0=extract_X_Y(fname, no_descriptor_names, y_column_name,smiles_column_name,fp_selection='maccs')
    X_train0, X_test0, y_train0, y_test0 = train_test_split(x0, y0, test_size=0.2, random_state=42)
        
    model0 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model0.fit(X_train0, y_train0)
    preds0 = model0.predict(X_test0)
    print(f"0 R^2 Score:{r2_score(y_test0, preds0):.3f} MAE:{mean_absolute_error(y_test0, preds0):.3f}")

    
    filename='rf_maccs_xScaffold.pkl'
    if not os.path.exists(filename):
        with open(filename,'wb') as f:
            pickle.dump(model0,f)

    if not os.path.exists("X_test.pkl"):
        with open("X_test.pkl",'wb') as f:
            pickle.dump(X_test0, f)

    if not os.path.exists("X_train.pkl"):
        with open("X_train.pkl",'wb') as f:
            pickle.dump(X_train0, f)

    if not os.path.exists("y_test.pkl"):
        with open("y_test.pkl",'wb') as f:
            pickle.dump(y_test0, f)

    if not os.path.exists("y_train.pkl"):
        with open("y_train.pkl",'wb') as f:
            pickle.dump(y_test0, f)




