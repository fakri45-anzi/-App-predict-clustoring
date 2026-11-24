from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"; MODELS=ROOT/"models"; MODELS.mkdir(exist_ok=True)
def main():
    datasets={
      "iris":(pd.read_csv(DATA/"iris.csv"),3,3),
      "wine":(pd.read_csv(DATA/"wine.csv"),3,3),
      "customer":(pd.read_csv(DATA/"Customer.csv"),3,2),
      # Le notebook Raisin ne définit pas de clustering. K=2 est retenu ici
      # car le dataset possède deux classes; Class reste exclue de l'entraînement.
      "raisin":(pd.read_excel(DATA/"Raisin.xlsx").drop(columns=["Class"],errors="ignore"),2,2)
    }
    for name,(df,kk,kc) in datasets.items():
        df=df.select_dtypes(include=[np.number]).copy()
        if df.empty or df.isna().any().any(): raise ValueError(f"{name}: données vides ou manquantes")
        norm=Normalizer(norm="l2"); X=norm.fit_transform(df)
        km=KMeans(n_clusters=kk,init="random",max_iter=500,random_state=42,n_init=20).fit(X)
        cah=AgglomerativeClustering(n_clusters=kc,linkage="average")
        labels=cah.fit_predict(X); classes=np.unique(labels)
        centers=np.vstack([X[labels==c].mean(axis=0) for c in classes])
        common={"dataset":name,"features":list(df.columns),"normalizer":norm}
        joblib.dump({**common,"model":km,"silhouette":float(silhouette_score(X,km.labels_))},MODELS/f"{name}_kmeans.joblib")
        joblib.dump({**common,"labels":labels,"centers":centers,"classes":classes,"silhouette":float(silhouette_score(X,labels))},MODELS/f"{name}_cah.joblib")
        print(f"{name}: modèles K-Means et CAH sauvegardés")
if __name__=="__main__": main()
