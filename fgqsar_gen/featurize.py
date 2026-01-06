#!/usr/bin/env python3
import numpy as np
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, AllChem, PandasTools, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

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



def generate_descriptors(mol,selected_descriptor_names):
    calculator=MoleculeDescriptors.MolecularDescriptorCalculator(selected_descriptor_names)
    return list(calculator.CalcDescriptors(mol))
