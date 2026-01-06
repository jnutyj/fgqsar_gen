#!/usr/bin/env python3
import numpy as np
import pandas as pd
from .generator import attach_fragments
from .qsar import RFQSAR
from rdkit import Chem



def design_molecules(scaffold,fg_dict, model_path, n_gen=300, topk=10):


    qsar = RFQSAR(model_path)
    cands,smis =  attach_fragments(scaffold, fg_dict,n=n_gen)
    valids,preds= qsar.predict_results(cands)
    top_hits = qsar.pick_top_hits(valids,preds,n=topk)

    results=[]

    for mol,p in top_hits:
        #print(smi)
        smi=Chem.MolToSmiles(mol)
        used_fgs = [name for name, fg in fg_dict.items() if fg.replace('[*]', '') in smi]
    results.append({
        "SMILES": smi,
        "Pred_pIC50": round(float(p), 2),
        "Design": ", ".join(used_fgs)
    })

    df_hits = pd.DataFrame(results)
    return df_hits
    


