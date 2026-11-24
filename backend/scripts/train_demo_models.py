from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

OUT=Path(__file__).resolve().parents[1]/"artifacts"
OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(42); n=4000
rain=rng.uniform(0,150,n); water=rng.uniform(0,2.5,n); drainage=rng.uniform(10,100,n); previous=rng.uniform(0,120,n)
latent=.42*(rain/150)+.30*(water/2.5)+.18*(1-drainage/100)+.10*(previous/120)
prob=1/(1+np.exp(-12*(latent-.48))); y=rng.binomial(1,prob)
X=pd.DataFrame({"rainfall_mm_hour":rain,"water_level_m":water,"drainage_capacity_pct":drainage,"previous_rainfall_mm":previous})
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=200,max_depth=9,random_state=42,class_weight="balanced").fit(Xtr,ytr)
pred=model.predict(Xte); p=model.predict_proba(Xte)[:,1]
print(classification_report(yte,pred,zero_division=0)); print("Synthetic-data ROC-AUC:",round(roc_auc_score(yte,p),4))
joblib.dump(model,OUT/"demo_flood_model.joblib")
print("Saved demo artifact. Synthetic data only; not a real flood model.")
