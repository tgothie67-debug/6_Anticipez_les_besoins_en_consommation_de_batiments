Seattle Building Energy Prediction
Contexte du projet

La ville de Seattle s'est fixée un objectif ambitieux : devenir neutre en émissions de carbone d'ici 2050.

Dans cette optique, la municipalité analyse la consommation énergétique et les émissions de CO₂ des bâtiments non résidentiels afin d’identifier les leviers d’amélioration énergétique.

Des relevés détaillés ont été réalisés en 2016 sur un ensemble de bâtiments. Cependant, ces mesures sont coûteuses à collecter, et il n'est pas possible de les obtenir pour tous les bâtiments.

L’objectif du projet est donc de :

Prédire la consommation énergétique totale d’un bâtiment
Prédire les émissions de CO₂
uniquement à partir de caractéristiques structurelles du bâtiment

Exemples de variables utilisées :

surface du bâtiment
nombre d’étages
type d’usage
année de construction
localisation géographique

Le projet consiste à développer un pipeline complet de machine learning, puis à déployer le modèle en production sous forme d’API dans le cloud.

Objectifs du projet

Les objectifs sont les suivants :

Réaliser une analyse exploratoire des données
Nettoyer et transformer les données
Tester plusieurs modèles supervisés
Identifier les variables les plus importantes
Construire un pipeline de prédiction
Déployer le modèle via une API BentoML
Conteneuriser le service avec Docker
Déployer le service sur AWS ECS Fargate
Architecture du projet
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
API REST (/predict)
Structure du projet
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
Analyse exploratoire des données

Une analyse exploratoire (EDA) a été réalisée afin de comprendre la structure des données et identifier les variables importantes.

Les analyses réalisées :

distribution des consommations énergétiques
corrélations entre variables
impact de la surface et de l’usage du bâtiment
influence de l’année de construction
analyse des outliers

Insights principaux :

les grands bâtiments consomment significativement plus
l’usage du bâtiment impacte fortement la consommation
l’année de construction influence les performances énergétiques
la localisation géographique peut jouer un rôle
Nettoyage et transformation des données

Plusieurs étapes de feature engineering ont été réalisées.

Création de nouvelles variables
Nombre d'usages du bâtiment
df['NumberOfPropertyUseTypes'] = df['ListOfAllPropertyUseTypes'].str.split(',').str.len()
Âge du bâtiment
df["BuildingAge"] = data_year - df["YearBuilt"]
Interaction âge / taille
df["Age_x_Size"] = df["BuildingAge"] * df["PropertyGFABuilding(s)"]
Densité de surface
df["Floor_density"] = df["PropertyGFABuilding(s)"] / df["NumberofFloors"]
Transformation logarithmique

Pour réduire l’asymétrie des distributions :

df["PropertyGFABuilding(s)_log"] = np.log(df["PropertyGFABuilding(s)"])
Encodage des variables catégorielles

One Hot Encoding :

pd.get_dummies(df, columns=["PrimaryPropertyType"])
Modèles de Machine Learning

Deux modèles ont été construits :

Modèle 1 : Consommation énergétique
RandomForestRegressor

Hyperparamètres :

n_estimators = 300
max_depth = 20
max_features = log2
min_samples_split = 5

Target :

SiteEnergyUse(kBtu)

Transformation :

log(target)
Modèle 2 : Émissions de CO₂
RandomForestRegressor

La prédiction de la consommation énergétique est utilisée comme feature supplémentaire.

Target :

TotalGHGEmissions
Sauvegarde des modèles avec BentoML

Les modèles sont enregistrés dans le Model Store BentoML.

Script :

train_and_save.py

Exemple :

bentoml.sklearn.save_model(
    "energy_rf_model",
    rf_model
)
Création du service API avec BentoML

Le fichier :

service.py

permet de créer une API de prédiction.

Endpoint principal :

POST /predict

Entrée JSON :

{
 "records": [
   {
     "YearBuilt": 1990,
     "NumberofFloors": 10
   }
 ]
}

Sortie :

[
 {
  "SiteEnergyUse(kBtu)_pred": 1537560,
  "TotalGHGEmissions_pred": 31.8
 }
]
Packaging du service BentoML

Commande :

python -m bentoml build

Cette commande :

lit bentofile.yaml
inclut les dépendances
inclut les modèles
crée un Bento package
Containerisation avec Docker

Commande :

python -m bentoml containerize buildingpredictionservice:latest

L’image Docker contient :

le modèle
le service API
les dépendances Python

Test local :

docker run -p 3000:3000 buildingpredictionservice:latest
Publication de l’image dans Amazon ECR

Création du repository :

aws ecr create-repository --repository-name buildingpredictionservice

Authentification Docker :

aws ecr get-login-password --region eu-west-3 \
| docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.eu-west-3.amazonaws.com

Tag de l’image :

docker tag buildingpredictionservice:latest \
ACCOUNT_ID.dkr.ecr.eu-west-3.amazonaws.com/buildingpredictionservice:latest

Push :

docker push ACCOUNT_ID.dkr.ecr.eu-west-3.amazonaws.com/buildingpredictionservice:latest
Déploiement sur AWS ECS Fargate

Création du cluster :

aws ecs create-cluster \
--cluster-name building-prediction-cluster
Création du rôle IAM
ecsTaskExecutionRole

Permet au conteneur :

tirer l’image depuis ECR
écrire dans CloudWatch Logs
Création du log group
aws logs create-log-group \
--log-group-name /ecs/building-prediction
Enregistrement de la task definition
aws ecs register-task-definition \
--cli-input-json file://task-definition.json
Création du service ECS
aws ecs create-service \
--cluster building-prediction-cluster \
--service-name building-prediction-service \
--task-definition building-prediction-task:1 \
--desired-count 1 \
--launch-type FARGATE
Accès au service

Une fois la tâche lancée, l’API est accessible via :

http://PUBLIC_IP:3000

Endpoint :

POST /predict

Interface Swagger :

http://PUBLIC_IP:3000
Gestion des coûts

Pour arrêter le service :

aws ecs update-service \
--cluster building-prediction-cluster \
--service building-prediction-service \
--desired-count 0

Cela stoppe les containers Fargate.

Technologies utilisées
Python
Pandas
Scikit-Learn
BentoML
Docker
AWS ECR
AWS ECS Fargate
CloudWatch
Résultat

Ce projet met en place un pipeline MLOps complet permettant :

d’entraîner un modèle
de le transformer en API
de le conteneuriser
de le déployer automatiquement dans le cloud

Le modèle est désormais consommable via une API REST.