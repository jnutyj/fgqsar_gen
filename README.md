# fgqsar_gen

FG-guided QSAR-based molecule generation for AIDD.

## Features
- RF + MACCS QSAR prediction
- SHAP-derived functional group guidance
- Scaffold-based generation using dummy atoms
- Returns top hits with design explanations

## Usage

```python
from fgqsar_gen.pipeline import design_molecules

fgs = {
    "Me": "[*]C",
    "Cl": "[*]Cl",
    "NH2": "[*]N",
    "OMe": "[*]OC"
}

scaffold = "c1ccc([*])cc1[*]"
df = design_molecules(scaffold, fgs, "models/rf_model.pkl")

print(df)
