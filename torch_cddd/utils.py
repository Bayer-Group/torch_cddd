from rdkit import Chem
from rdkit.Chem import Descriptors


def transform_smiles(smiles, smiles_type):
    mol = Chem.MolFromSmiles(smiles)
    if smiles_type == "random":
        smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    elif smiles_type == "canonical":
        smiles = Chem.MolToSmiles(mol, doRandom=False, canonical=True)
    return smiles


def get_descriptors(sml):
    m = Chem.MolFromSmiles(sml)
    descriptor_list = [
        Descriptors.MolLogP(m),
        Descriptors.MolMR(m),
        Descriptors.BalabanJ(m),
        Descriptors.NumHAcceptors(m),
        Descriptors.NumHDonors(m),
        Descriptors.NumValenceElectrons(m),
        Descriptors.TPSA(m)
    ]
    return descriptor_list