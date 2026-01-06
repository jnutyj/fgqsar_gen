#!/usr/bin/env python3
from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_main_scaffold(smiles_list):
    scaffold_smiles_list=[]
    for smi in smiles_list:
        # mol=Chem.MolFromSmiles(smi)
        # if mol:
        #     scaffold=MurckoScaffold.MurckoDecompose(mol)
        #     scaffold_smiles=Chem.MolToSmiles(scaffold, canonical=True)
        #     scaffold_smiles_list.append(scaffold_smiles)
        scaffold_smiles= MurckoScaffold.MurckoScaffoldSmiles(smi)
        scaffold_smiles_list.append(scaffold_smiles)

    scaffold_counts=Counter(scaffold_smiles_list)
    main_scaffold_smi, count = scaffold_counts.most_common(1)[0]
    return main_scaffold_smi, count



def get_scaffold(smiles):
    scaffold_smiles= MurckoScaffold.MurckoScaffoldSmiles(smiles)
    return scaffold_smiles

