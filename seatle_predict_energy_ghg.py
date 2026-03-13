# Librairies
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Import des données brutes du/des bâtiment(s) à prédire (ici données d'exemple)
df_new_building = pd.read_excel("data/buildings_to_predict.xlsx")

# On conserve l'ID à part
building_ids = df_new_building["OSEBuildingID"].copy()


### FEATURE ENGINEERING ###

# Nombre de type d'usage dans un bâtiment
df_new_building['NumberOfPropertyUseTypes'] = df_new_building['ListOfAllPropertyUseTypes'].str.split(',').str.len()
df_new_building = df_new_building.drop(columns=['ListOfAllPropertyUseTypes'])

# Ajout de l'âge du bâtiment
data_year = df_new_building['DataYear'].drop_duplicates().iloc[0]
df_new_building["BuildingAge"] = data_year - df_new_building["YearBuilt"]

# Croissement entre l'âge et la surface (grand et ancien = plus énergivore ?)
df_new_building["Age_x_Size"] = df_new_building["BuildingAge"] * df_new_building["PropertyGFABuilding(s)"]

# Surface par étage
df_new_building["Floor_density"] = df_new_building["PropertyGFABuilding(s)"] / df_new_building["NumberofFloors"]

# Récupère la surface moyenne par type d'usage principal du bâtiment depuis les données d'entrainement
train_raw = pd.read_csv("data/features.csv")
mapping_size = (train_raw[["LargestPropertyUseType", "size_x_type_mean"]].drop_duplicates())
df_new_building = df_new_building.merge(
    mapping_size,
    on="LargestPropertyUseType",
    how="left"
)

# One Hot Encoding sur l'usage principal
df_new_building = pd.get_dummies(df_new_building, columns=["PrimaryPropertyType"])

# Transformation logarithmique sur la colonne 'PropertyGFABuilding(s)' et 'Age_x_Size' pour éviter une distribution asymétrique
df_new_building["PropertyGFABuilding(s)_log"] = np.log(df_new_building["PropertyGFABuilding(s)"])
df_new_building["Age_x_Size_log"] = np.log(df_new_building["Age_x_Size"])

features = [
    'NumberofFloors',
    'PropertyGFABuilding(s)_log',
    'LargestPropertyUseTypeGFA',
    'Latitude',
    'Longitude',
    'NumberOfPropertyUseTypes',
    'BuildingAge',
    'Age_x_Size_log',
    'Floor_density',
    'size_x_type_mean',
    'PrimaryPropertyType_Distribution Center',
    'PrimaryPropertyType_Hotel',
    'PrimaryPropertyType_K-12 School',
    'PrimaryPropertyType_Large Office',
    'PrimaryPropertyType_Low-Rise Multifamily',
    'PrimaryPropertyType_Medical Office',
    'PrimaryPropertyType_Mixed Use Property',
    'PrimaryPropertyType_Other',
    'PrimaryPropertyType_Refrigerated Warehouse',
    'PrimaryPropertyType_Residence Hall',
    'PrimaryPropertyType_Restaurant',
    'PrimaryPropertyType_Retail Store',
    'PrimaryPropertyType_Self-Storage Facility',
    'PrimaryPropertyType_Senior Care Community',
    'PrimaryPropertyType_Small- and Mid-Sized Office',
    'PrimaryPropertyType_Supermarket / Grocery Store',
    'PrimaryPropertyType_University',
    'PrimaryPropertyType_Warehouse',
    'PrimaryPropertyType_Worship Facility'
]

# Ajouter les colonnes manquantes avec la valeur False
for colonne in features:
    if colonne not in df_new_building.columns:
        df_new_building[colonne] = False

# Sélectionner les colonnes
df_new_building_features = df_new_building[features]

###########################


### MODÈLE ML, PRÉDIRE LA CONSOMMATION ÉNERGÉTIQUE: RANDOM FOREST ###

X = pd.read_csv("data/features.csv")
X = X.drop(columns=['SiteEnergyUse(kBtu)', 'LargestPropertyUseType'])

y = pd.read_csv("data/energy_target.csv").squeeze()

# Transformation logarithmique sur la target
y = np.log(y)

# Grille d’hyperparamètres
rf_final = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    max_features="log2",
    min_samples_split=5,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Entraîner le modèle
rf_final.fit(X, y)

# Prédiction sur les données
y_pred_log = rf_final.predict(df_new_building_features)

# Retour à l'échelle normale
y_pred_energy = np.exp(y_pred_log)


### MODÈLE ML, PRÉDIRE L'ÉMISSION DE CO2: RANDOM FOREST ###

X = pd.read_csv("data/features.csv")
X = X.drop(columns=['LargestPropertyUseType'])

y = pd.read_csv("data/ghg_target.csv").squeeze()

# Ajouter la consommation d'énergie prédite pour le modèle
df_new_building_features["SiteEnergyUse(kBtu)"] = y_pred_energy

# Grille d’hyperparamètres
rf_final = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    max_features=0.5,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Entraîner le modèle
rf_final.fit(X, y)

# Prédiction sur les données
y_pred_ghg = rf_final.predict(df_new_building_features)

###########################
# AFFICHAGE DES RÉSULTATS
###########################

results = pd.DataFrame({
    "OSEBuildingID": building_ids,
    "SiteEnergyUse(kBtu)_pred": y_pred_energy,
    "TotalGHGEmissions_pred": y_pred_ghg
})

print(results)