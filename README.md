# Clustering Studio — 4 datasets

Installation: `py -m pip install -r requirements.txt`

Entraînement et sauvegarde des 8 artefacts: `py train_models.py`

Lancement: `py -m streamlit run app.py`

Deux interfaces seulement: prédiction simple et prédiction multiple CSV/Excel.

Customer reprend le notebook: normalisation L2, K-Means K=3, CAH average K=2.
Raisin: le notebook fourni fait normalisation/ACP mais ne définit pas de clustering.
K=2 est donc un choix de modélisation explicite, cohérent avec les deux classes
du fichier; la colonne Class est exclue de l'entraînement.
Iris et Wine utilisent K=3. CAH n'a pas de predict natif; affectation au centroïde
de groupe le plus proche pour les nouvelles observations.
