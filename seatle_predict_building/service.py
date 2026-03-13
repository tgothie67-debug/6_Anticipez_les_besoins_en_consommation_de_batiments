from __future__ import annotations

import pandas as pd
import numpy as np
import bentoml
from pydantic import BaseModel
from typing import List, Dict, Any


# Chargement des modèles depuis le Model Store BentoML
energy_model = bentoml.sklearn.load_model("energy_rf_model:latest")
ghg_model = bentoml.sklearn.load_model("ghg_rf_model:latest")


# Features attendues par les modèles
ENERGY_FEATURES = [
    "NumberofFloors",
    "PropertyGFABuilding(s)_log",
    "LargestPropertyUseTypeGFA",
    "Latitude",
    "Longitude",
    "NumberOfPropertyUseTypes",
    "BuildingAge",
    "Age_x_Size_log",
    "Floor_density",
    "size_x_type_mean",
    "PrimaryPropertyType_Distribution Center",
    "PrimaryPropertyType_Hotel",
    "PrimaryPropertyType_K-12 School",
    "PrimaryPropertyType_Large Office",
    "PrimaryPropertyType_Low-Rise Multifamily",
    "PrimaryPropertyType_Medical Office",
    "PrimaryPropertyType_Mixed Use Property",
    "PrimaryPropertyType_Other",
    "PrimaryPropertyType_Refrigerated Warehouse",
    "PrimaryPropertyType_Residence Hall",
    "PrimaryPropertyType_Restaurant",
    "PrimaryPropertyType_Retail Store",
    "PrimaryPropertyType_Self-Storage Facility",
    "PrimaryPropertyType_Senior Care Community",
    "PrimaryPropertyType_Small- and Mid-Sized Office",
    "PrimaryPropertyType_Supermarket / Grocery Store",
    "PrimaryPropertyType_University",
    "PrimaryPropertyType_Warehouse",
    "PrimaryPropertyType_Worship Facility"
]

GHG_FEATURES = ENERGY_FEATURES + ["SiteEnergyUse(kBtu)"]


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


def feature_engineering(df_new_building: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    df_new_building = df_new_building.copy()

    building_ids = df_new_building["OSEBuildingID"].copy()

    # Nombre de types d'usage
    df_new_building["NumberOfPropertyUseTypes"] = (
        df_new_building["ListOfAllPropertyUseTypes"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .str.len()
    )
    df_new_building = df_new_building.drop(columns=["ListOfAllPropertyUseTypes"])

    # Âge du bâtiment
    data_year = df_new_building["DataYear"].drop_duplicates().iloc[0]
    df_new_building["BuildingAge"] = data_year - df_new_building["YearBuilt"]

    # Croisement âge x surface
    df_new_building["Age_x_Size"] = (
        df_new_building["BuildingAge"] * df_new_building["PropertyGFABuilding(s)"]
    )

    # Densité par étage
    df_new_building["Floor_density"] = (
        df_new_building["PropertyGFABuilding(s)"] / df_new_building["NumberofFloors"]
    )

    # Mapping depuis les données train
    train_raw = pd.read_csv("data/features.csv")
    mapping_size = (
        train_raw[["LargestPropertyUseType", "size_x_type_mean"]]
        .drop_duplicates()
    )

    df_new_building = df_new_building.merge(
        mapping_size,
        on="LargestPropertyUseType",
        how="left"
    )

    # One-hot encoding
    df_new_building = pd.get_dummies(
        df_new_building,
        columns=["PrimaryPropertyType"]
    )

    # Sécurisation des logs
    df_new_building["PropertyGFABuilding(s)_log"] = np.log(
        df_new_building["PropertyGFABuilding(s)"].clip(lower=1e-9)
    )
    df_new_building["Age_x_Size_log"] = np.log(
        df_new_building["Age_x_Size"].clip(lower=1e-9)
    )

    # Colonnes manquantes
    for col in ENERGY_FEATURES:
        if col not in df_new_building.columns:
            df_new_building[col] = 0

    df_features = df_new_building[ENERGY_FEATURES].copy()

    return building_ids, df_features


@bentoml.service(
    traffic={"timeout": 60}
)
class BuildingPredictionService:

    @bentoml.api
    def predict(self, input_data: PredictRequest) -> List[Dict[str, Any]]:
        df_new_building = pd.DataFrame(input_data.records)

        building_ids, df_features = feature_engineering(df_new_building)

        # 1. prédiction énergie
        y_pred_log = energy_model.predict(df_features)
        y_pred_energy = np.exp(y_pred_log)

        # 2. prédiction GHG
        df_features_ghg = df_features.copy()
        df_features_ghg["SiteEnergyUse(kBtu)"] = y_pred_energy

        y_pred_ghg = ghg_model.predict(df_features_ghg)

        results = pd.DataFrame({
            "OSEBuildingID": building_ids,
            "SiteEnergyUse(kBtu)_pred": y_pred_energy,
            "TotalGHGEmissions_pred": y_pred_ghg
        })

        return results.to_dict(orient="records")