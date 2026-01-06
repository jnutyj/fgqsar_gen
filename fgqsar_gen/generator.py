#!/usr/bin/env python3
from rdkit import Chem
import random

def connect_on_dummy(mol, frag):
    mol = Chem.RWMol(mol)
    frag = Chem.RWMol(frag)

    # find dummy
    mol_dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    frag_dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetSymbol() == '*']
    if not mol_dummies or not frag_dummies:
        return None

    m_idx = mol_dummies[0]
    f_idx = frag_dummies[0]

    # dummy nearby
    m_nbr = mol.GetAtomWithIdx(m_idx).GetNeighbors()[0].GetIdx()
    f_nbr = frag.GetAtomWithIdx(f_idx).GetNeighbors()[0].GetIdx()

    combo = Chem.CombineMols(mol, frag)
    combo = Chem.RWMol(combo)

    offset = mol.GetNumAtoms()
    ### in this case, it can only recognize the single bond, rather than other types of bonds
    ## TODO build another function to make it more generalize for different cases
    combo.AddBond(m_nbr, f_nbr + offset, Chem.BondType.SINGLE)

    # delete dummy
    combo.RemoveAtom(f_idx + offset)
    combo.RemoveAtom(m_idx)

    try:
        Chem.SanitizeMol(combo)
        return combo
    except:
        return None


def attach_fragments(scaffold_smi, frag_smis, n=200):
    mols = []
    smis = []

    for _ in range(n):
        mol = Chem.MolFromSmiles(scaffold_smi)
        if mol is None:
            continue

        for _ in range(5):
            if '*' not in Chem.MolToSmiles(mol):
                break
            frag = Chem.MolFromSmiles(random.choice(list(frag_smis.values())))
            new_mol = connect_on_dummy(mol, frag)
            if new_mol is None:
                break
            mol = new_mol

        smi = Chem.MolToSmiles(mol)
        if '*' not in smi:
            mols.append(mol)
            smis.append(smi)

    return list(set(mols)),smis
