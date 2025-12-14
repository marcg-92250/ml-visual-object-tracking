# MLVOT - Multi-Object Visual Tracking

Projet de suivi d'objets visuels multiples (MLVOT) implémentant différentes méthodes de tracking progressives, du suivi d'un seul objet avec filtre de Kalman à un système combinant apparence (ReID), géométrie (IoU) et prédiction de mouvement (Kalman).

## Structure du Projet

```
ml-visual-object-tracking/
├── tp1/                    # TP1: Kalman Filter 2D (Single Object)
│   ├── detector.py         # Détecteur d'objets (Canny edge detection)
│   ├── kalman_filter.py    # Implémentation du filtre de Kalman
│   └── obj_tracking.py     # Script principal de tracking
│
├── tp2/                    # TP2: IoU Tracker (Multiple Objects)
│   ├── utils/
│   │   ├── data_loader.py      # Chargement des détections MOT Challenge
│   │   └── visualization.py    # Fonctions de visualisation
│   ├── iou_tracker.py      # Tracker basé sur IoU + Hungarian algorithm
│   └── main.py             # Script principal
│
├── tp3/                    # TP3: Kalman + IoU Tracker
│   ├── kalman_filter.py    # Filtre de Kalman adapté aux bounding boxes
│   ├── kalman_iou_tracker.py  # Tracker combinant Kalman + IoU
│   └── main.py             # Script principal
│
├── tp4/                    # TP4: ReID + Kalman + IoU Tracker
│   ├── kalman_filter.py    # Filtre de Kalman (réutilisé)
│   ├── reid_extractor.py   # Extracteur de features ReID
│   ├── reid_tracker.py     # Tracker combinant IoU + Kalman + ReID
│   ├── main.py             # Script principal
│   └── evaluate.py         # Évaluation avec métriques MOT
│
├── ADL-Rundle-6/           # Dataset MOT Challenge
│   ├── img1/               # Images de la séquence (525 frames)
│   ├── det/                # Détections pré-générées
│   │   └── Yolov5l/
│   │       └── det.txt
│   └── gt/                 # Ground truth pour évaluation
│       └── gt.txt
│
├── 2D_Kalman-Filter_TP1/   # Données pour TP1
│   └── video/
│       └── randomball.avi  # Vidéo de test pour suivi d'un objet
│
├── reid_osnet_x025_market1501.onnx  # Modèle ReID pré-entraîné
├── requirements.txt        # Dépendances Python
├── RAPPORT.md             # Rapport détaillé du projet
└── README.md              # Ce fichier
```

## Installation

### Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des Dépendances

1. Clonez ou téléchargez le repository

2. Installez les dépendances Python :
```bash
pip install -r requirements.txt
```

Ou installez manuellement :
```bash
pip install numpy opencv-python scipy onnxruntime
```

Les dépendances incluent :
- **numpy** : Calculs numériques et manipulation de matrices
- **opencv-python** : Traitement d'images et vidéos
- **scipy** : Algorithme hongrois (linear_sum_assignment)
- **onnxruntime** : Inférence du modèle ReID ONNX (requis pour TP4)

## Utilisation

### TP1: Single Object Tracking with Kalman Filter

Suivi d'un seul objet (balle) avec filtre de Kalman.

```bash
cd tp1
python obj_tracking.py
```

**Paramètres du Kalman Filter** (modifiables dans `obj_tracking.py`) :
- `dt=0.1` : Pas de temps
- `u_x=1, u_y=1` : Accélérations
- `std_acc=1` : Bruit de processus
- `x_std_meas=0.1, y_std_meas=0.1` : Bruit de mesure

**Visualisation** :
- Cercle vert : Détection
- Rectangle bleu : Prédiction
- Rectangle rouge : Estimation
- Ligne jaune : Trajectoire

### TP2: IoU-based Tracking

Suivi multi-objets basé sur l'Intersection over Union (IoU).

```bash
cd tp2
python main.py
```

**Paramètres** (modifiables dans `main.py`) :
- `iou_threshold=0.3` : Seuil IoU minimum pour un match
- `max_age=5` : Nombre de frames avant suppression d'un track non matché
- `conf_threshold=0.7` : Seuil de confiance pour les détections

**Sorties** :
- `ADL-Rundle-6.txt` : Résultats au format MOT Challenge
- `output.mp4` : Vidéo avec tracking visualisé

### TP3: Kalman-Guided IoU Tracking

Combinaison du filtre de Kalman avec l'association IoU pour une meilleure robustesse.

```bash
cd tp3
python main.py
```

**Améliorations par rapport au TP2** :
- Prédiction de mouvement avec Kalman Filter
- Association basée sur la position prédite
- Meilleure gestion des occlusions temporaires (`max_age=10`)

**Sorties** :
- `ADL-Rundle-6.txt` : Résultats au format MOT Challenge
- `output.mp4` : Vidéo avec tracking visualisé

### TP4: Appearance-Aware IoU-Kalman Tracker

Ajout de features ReID (Re-Identification) pour améliorer l'association.

```bash
cd tp4
python main.py
```

**Paramètres** (modifiables dans `main.py`) :
- `iou_threshold=0.3` : Seuil pour le score combiné
- `max_age=30` : Nombre de frames avant suppression
- `alpha=0.6` : Poids pour IoU dans le score combiné
- `beta=0.4` : Poids pour la similarité ReID

**Score combiné** :
```
S = α * IoU + β * Normalized_Similarity
```

où `Normalized_Similarity = 1 / (1 + Euclidean_Distance)`

**Sorties** :
- `ADL-Rundle-6.txt` : Résultats au format MOT Challenge
- `output.mp4` : Vidéo avec tracking visualisé

### Évaluation (TP4)

Calculer les métriques de performance :

```bash
cd tp4
python evaluate.py
```

**Métriques calculées** :
- **MOTA** : Multiple Object Tracking Accuracy
- **IDF1** : F1 score pour la préservation des IDs
- **Precision** : Précision des détections
- **Recall** : Rappel des détections
- **ID_Switches** : Nombre de changements d'ID
- **FPS** : Frames par seconde

## Résultats et Vidéos

### Emplacement des Vidéos de Résultats

Après l'exécution des scripts, les vidéos de visualisation sont générées dans les dossiers respectifs :

- **TP2** : `tp2/output.mp4` - Résultats du tracker IoU
- **TP3** : `tp3/output.mp4` - Résultats du tracker Kalman-IoU
- **TP4** : `tp4/output.mp4` - Résultats du tracker ReID-Kalman-IoU

Ces vidéos montrent les bounding boxes colorées pour chaque track, avec les IDs affichés, permettant de visualiser la continuité des identités et la qualité du tracking.

**Note importante** : Les fichiers vidéo (`*.mp4`) et les résultats (`*.txt`) ne sont pas inclus dans le repository Git car ils sont trop volumineux. Ils sont générés localement lors de l'exécution des scripts.

### Fichiers de Résultats

Les fichiers de résultats au format MOT Challenge sont également générés :

- `tp2/ADL-Rundle-6.txt` : Résultats du TP2
- `tp3/ADL-Rundle-6.txt` : Résultats du TP3
- `tp4/ADL-Rundle-6.txt` : Résultats du TP4

Ces fichiers peuvent être utilisés pour l'évaluation avec des outils comme TrackEval.

## Format des Données

### Détections (MOT Challenge format)

Fichier `det.txt` avec format :
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

### Résultats de tracking

Fichier de sortie au même format, avec les IDs assignés par le tracker.

### Ground Truth

Fichier `gt.txt` pour l'évaluation, même format que les détections.

## Notes Techniques

### TP1
- Détecteur basé sur Canny edge detection
- Filtre de Kalman avec état [x, y, vx, vy]
- Modèle de mouvement constant avec accélération

### TP2
- Association par Hungarian algorithm (scipy.optimize.linear_sum_assignment)
- Gestion des tracks : création, mise à jour, suppression
- Calcul IoU entre bounding boxes

### TP3
- Kalman Filter sur centroïdes des bounding boxes
- Dimensions (largeur/hauteur) conservées séparément
- Prédiction avant association pour meilleure robustesse

### TP4
- Modèle ReID : OSNet pré-entraîné sur Market1501
- Features 512-dim normalisées L2
- Preprocessing : resize 64x128, BGR→RGB, normalisation ImageNet
- Distance euclidienne entre features pour similarité

## Exécution Complète

Pour exécuter tous les TPs dans l'ordre :

```bash
# TP1
cd tp1 && python obj_tracking.py

# TP2
cd ../tp2 && python main.py

# TP3
cd ../tp3 && python main.py

# TP4
cd ../tp4 && python main.py

# Évaluation TP4
cd ../tp4 && python evaluate.py
```


