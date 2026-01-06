#!/usr/bin/env python3
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem


def draw_highlight(mol, atoms):
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=atoms)
    drawer.FinishDrawing()
    return SVG(drawer.GetDrawingText())


def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit molecule object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        # Optional: Sanitize molecule, add H, etc. as needed
        # Chem.SanitizeMol(mol)
        return mol
    except:
        return None                                                                                                
def get_highlight_atoms(mol, smarts_list):
    highlight_atoms = set()
    for smarts in smarts_list:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                highlight_atoms.update(match)
    return list(highlight_atoms)
