import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

# ===============================
# Load Trained Model
# ===============================
model = joblib.load("drug_legality_model.pkl")


# ===============================
# Feature Extraction Function
# ===============================
def extract_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Molecular descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)

    # Morgan Fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_array = np.array(fp)

    return np.concatenate(([mw, logp, h_donors, h_acceptors, tpsa], fp_array))


# ===============================
# STREAMLIT UI
# ===============================

st.title("💊 Drug Legal / Illegal Classifier")

st.write("Enter Drug Name and SMILES to classify drug legality")

# Input Fields
drug_name = st.text_input("Enter Drug Name")
smiles = st.text_input("Enter SMILES String")


# Predict Button
if st.button("Predict Classification"):

    if smiles.strip() == "":
        st.warning("⚠ Please enter SMILES string")

    else:
        features = extract_features(smiles)

        if features is None:
            st.error("❌ Invalid SMILES string")

        else:
            prediction = model.predict(features.reshape(1, -1))

            st.success(f"✅ Prediction: {prediction[0]}")

