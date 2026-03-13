import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import bentoml


def train_energy_model():
    X = pd.read_csv("data/features.csv")
    X = X.drop(columns=["SiteEnergyUse(kBtu)", "LargestPropertyUseType"])

    y = pd.read_csv("data/energy_target.csv").squeeze()
    y = np.log(y)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        max_features="log2",
        min_samples_split=5,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    bentoml.sklearn.save_model(
        "energy_rf_model",
        model,
        metadata={
            "target": "SiteEnergyUse(kBtu)",
            "log_target": True
        }
    )

    print("Modèle énergie sauvegardé dans BentoML Model Store.")


def train_ghg_model():
    X = pd.read_csv("data/features.csv")
    X = X.drop(columns=["LargestPropertyUseType"])

    y = pd.read_csv("data/ghg_target.csv").squeeze()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        max_features=0.5,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    bentoml.sklearn.save_model(
        "ghg_rf_model",
        model,
        metadata={
            "target": "TotalGHGEmissions",
            "uses_energy_prediction": True
        }
    )

    print("Modèle GHG sauvegardé dans BentoML Model Store.")


if __name__ == "__main__":
    train_energy_model()
    train_ghg_model()