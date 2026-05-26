"""Train 5 classifiers on Wisconsin Diagnostic with 5-fold CV and calibration."""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

DATA = "data/wisconsin"
OUT  = "data/models"
os.makedirs(OUT, exist_ok=True)

X_train = pd.read_csv(f"{DATA}/X_train.csv").values
y_train = pd.read_csv(f"{DATA}/y_train.csv").values.ravel()
X_test  = pd.read_csv(f"{DATA}/X_test.csv").values
y_test  = pd.read_csv(f"{DATA}/y_test.csv").values.ravel()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_models = {
    "LR":  LogisticRegression(max_iter=1000, random_state=42),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "NB":  GaussianNB(),
    "XGB": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0),
}

for name, model in base_models.items():
    model.fit(X_train, y_train)
    platt    = CalibratedClassifierCV(model, method="sigmoid",  cv="prefit")
    isotonic = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    platt.fit(X_train, y_train)
    isotonic.fit(X_train, y_train)

    joblib.dump(model,    f"{OUT}/{name}_base.pkl")
    joblib.dump(platt,    f"{OUT}/{name}_platt.pkl")
    joblib.dump(isotonic, f"{OUT}/{name}_isotonic.pkl")
    print(f"Saved {name} (base, platt, isotonic)")
