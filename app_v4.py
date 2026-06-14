
import streamlit as st
import numpy as np, joblib, json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, DataStructs

st.set_page_config(page_title="Drug Toxicity Predictor v4",
                   page_icon="🧪", layout="wide")

@st.cache_resource
def load_artifacts():
    valid_tasks  = joblib.load("saved_models/mt_valid_tasks.pkl")
    mt_scaler    = joblib.load("saved_models/mt_scaler.pkl")
    desc_names   = joblib.load("saved_models/NR-AR_desc_names.pkl")
    mt_thresholds= json.load(open("saved_models/mt_thresholds.json"))
    mt_models    = {}
    for t in valid_tasks:
        try: mt_models[t] = joblib.load(f"saved_models/mt_{t}_xgb.pkl")
        except: pass
    return mt_models, mt_scaler, desc_names, mt_thresholds, valid_tasks

def smiles_to_features(smiles, scaler, desc_names):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp  = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), fp)
    fn_map = dict(Descriptors.descList)
    dv = []
    for nm in desc_names:
        try: dv.append(float(fn_map[nm](mol) or 0.0))
        except: dv.append(0.0)
    X_desc = np.nan_to_num(np.array(dv).reshape(1,-1))
    return np.hstack([fp.reshape(1,-1), scaler.transform(X_desc)])

def main():
    mt_models, mt_scaler, desc_names, mt_thresholds, valid_tasks = load_artifacts()

    st.title("🧪 Drug Toxicity Predictor v4.0")
    st.caption("Multi-Task Tox21 · SMOTE + 5-Fold StratifiedKFold · Bug-free rebuild")

    col1, col2 = st.columns([2,1])
    with col1:
        smiles = st.text_input("SMILES Senyawa", value="CC(=O)Oc1ccccc1C(=O)O",
                                help="Contoh: CC(=O)Oc1ccccc1C(=O)O (aspirin)")
        target = st.selectbox("Target Tox21", valid_tasks)

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    with col2:
        if mol:
            st.image(Draw.MolToImage(mol, size=(200,160)), caption="Struktur 2D")
        else:
            st.error("SMILES tidak valid")

    if st.button("🔬 Analisis Toksisitas", type="primary") and mol:
        X = smiles_to_features(smiles, mt_scaler, desc_names)
        if X is not None and target in mt_models:
            prob   = float(mt_models[target].predict_proba(X)[0][1])
            thresh = mt_thresholds.get(target, 0.5)
            is_tox = prob >= thresh
            if is_tox:
                st.error(f"⚠️ TOKSIK — Prob: {prob:.1%}  Threshold: {thresh:.3f}")
            else:
                st.success(f"✅ TIDAK TOKSIK — Prob: {prob:.1%}  Threshold: {thresh:.3f}")
            st.metric("Probabilitas Toksik", f"{prob:.1%}")
            st.metric("Threshold Optimal", f"{thresh:.3f}")
        else:
            st.warning("Fitur gagal diekstrak atau model tidak tersedia")

    st.caption("⚠️ Hanya untuk riset. Bukan pengganti uji lab atau diagnosis medis.")

if __name__ == "__main__":
    main()
