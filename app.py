import io
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
ROOT=Path(__file__).resolve().parent; MODELS=ROOT/"models"; DATA=ROOT/"data"
st.set_page_config(page_title="Clustering Studio",page_icon="🧩",layout="wide")
DATASETS={"Iris":"iris","Wine":"wine","Customer":"customer","Raisin":"raisin"}
@st.cache_resource
def load_artifact(key,method):
    p=MODELS/f"{key}_{'kmeans' if method=='K-Means' else 'cah'}.joblib"
    if not p.exists(): raise FileNotFoundError(f"{p.name} absent. Exécute d'abord : py train_models.py")
    return joblib.load(p)
def predict(frame,a,method):
    cols=a["features"]; xraw=frame[cols].apply(pd.to_numeric,errors="coerce")
    if xraw.isna().any().any(): raise ValueError("Valeurs manquantes ou non numériques.")
    X=a["normalizer"].transform(xraw)
    if method=="K-Means": y=a["model"].predict(X)
    else:
        d=((X[:,None,:]-a["centers"][None,:,:])**2).sum(axis=2)
        y=a["classes"][d.argmin(axis=1)]
    return y.astype(int)+1
st.title(" Clustering Studio")
st.caption("Prédiction de clusters — deux interfaces uniquement")
with st.sidebar:
    st.header("Configuration")
    label=st.selectbox("Dataset",list(DATASETS))
    key=DATASETS[label]
    method=st.selectbox("Méthode",["K-Means","CAH"])
try: a=load_artifact(key,method)
except Exception as e: st.error(str(e)); st.stop()
features=a["features"]
if key=="raisin": source=pd.read_excel(DATA/"Raisin.xlsx")
elif key=="customer": source=pd.read_csv(DATA/"Customer.csv")
else: source=pd.read_csv(DATA/f"{key}.csv")
med=source[features].median(numeric_only=True)
x1,x2,x3=st.columns(3); x1.metric("Variables",len(features)); x2.metric("Méthode",method); x3.metric("Silhouette entraînement",f"{a['silhouette']:.3f}")
t1,t2=st.tabs(["🔹 Prédiction simple"," Prédiction multiple"])
with t1:
    st.subheader("Prédire une observation")
    with st.form("simple"):
        cols=st.columns(2); vals={}
        for i,f in enumerate(features):
            vals[f]=cols[i%2].number_input(f,value=float(med[f]),format="%.6f",key=f"{key}_{f}")
        go=st.form_submit_button("Prédire le cluster",type="primary")
    if go:
        row=pd.DataFrame([[vals[f] for f in features]],columns=features)
        st.success(f"Observation affectée au **cluster {predict(row,a,method)[0]}**.")
with t2:
    st.subheader("Prédiction de plusieurs observations")
    st.caption("Colonnes attendues : "+" · ".join(features))
    up=st.file_uploader("Importer un CSV ou Excel",type=["csv","xlsx","xls"])
    if up:
        try:
            df=pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
            missing=[f for f in features if f not in df.columns]
            if missing: st.error("Colonnes absentes : "+", ".join(missing))
            else:
                out=df.copy(); out["cluster"]=predict(df,a,method)
                st.success(f"{len(out)} observation(s) traitée(s)."); st.dataframe(out,use_container_width=True)
                st.download_button("Télécharger CSV",out.to_csv(index=False).encode("utf-8-sig"),"predictions_clusters.csv","text/csv")
                buf=io.BytesIO()
                with pd.ExcelWriter(buf,engine="openpyxl") as w: out.to_excel(w,index=False)
                st.download_button("Télécharger Excel",buf.getvalue(),"predictions_clusters.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e: st.error(f"Erreur : {e}")
st.divider()
st.caption("La CAH n'a pas de predict natif : les nouvelles observations sont affectées au centroïde CAH le plus proche.")
