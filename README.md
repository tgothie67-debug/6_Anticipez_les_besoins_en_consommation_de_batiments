## Seattle Building Energy Prediction

# Contexte du projet

La ville de Seattle s'est fixée un objectif ambitieux : devenir neutre en émissions de carbone d'ici 2050.

Dans cette optique, la municipalité analyse la consommation énergétique et les émissions de CO₂ des bâtiments non résidentiels afin d’identifier les leviers d’amélioration énergétique.

Des relevés détaillés ont été réalisés en 2016 sur un ensemble de bâtiments. Cependant, ces mesures sont coûteuses à collecter, et il n'est pas possible de les obtenir pour tous les bâtiments.

L’objectif du projet est donc de :

- Prédire la consommation énergétique totale d’un bâtiment
- Prédire les émissions de CO₂

*uniquement à partir de caractéristiques structurelles du bâtiment*

# Exemples de variables utilisées :

- surface du bâtiment
- nombre d’étages
- type d’usage
- année de construction
- localisation géographique

# Objectifs du projet

Le projet consiste à développer un pipeline complet de machine learning, puis à déployer le modèle en production sous forme d’API dans le cloud.

- Réaliser une analyse exploratoire des données
- Nettoyer et transformer les données
- Tester plusieurs modèles supervisés
- Identifier les variables les plus importantes
- Construire un pipeline de prédiction
- Déployer le modèle via une API BentoML
- Conteneuriser le service avec Docker
- Déployer le service sur AWS ECS Fargate

# Structure du projet :
```bash
project
│
├── data
│   ├── features.csv
│   ├── energy_target.csv
│   └── ghg_target.csv
│
├── notebooks
│   └── exploration.ipynb
│
├── train_and_save.py
├── service.py
├── bentofile.yaml
│
├── payload.json
├── test_api.py
│
└── README.md
```

# Architecture du projet :

```bash
Raw Data
   │
   ▼
Data Cleaning & Feature Engineering
   │
   ▼
Machine Learning Models
   │
   ▼
BentoML API Service
   │
   ▼
Docker Container
   │
   ▼
Amazon ECR
   │
   ▼
AWS ECS Fargate
   │
   ▼
API REST --> http://35.180.125.72:3000/
```


Exemple d'entrée JSON afin d'utiliser le modèle :
```bash
{
  "input_data": {
    "records": [
      {
        "OSEBuildingID": 1,
        "DataYear": 2016,
        "YearBuilt": 1990,
        "ListOfAllPropertyUseTypes": "Office,Parking",
        "LargestPropertyUseType": "Office",
        "LargestPropertyUseTypeGFA": 25000,
        "PropertyGFABuilding(s)": 30000,
        "NumberofFloors": 10,
        "Latitude": 47.61,
        "Longitude": -122.33,
        "PrimaryPropertyType": "Large Office"
      }
    ]
  }
}
```

Sortie :
```bash
[
 {
  "SiteEnergyUse(kBtu)_pred": 1537560,
  "TotalGHGEmissions_pred": 31.8
 }
]
```