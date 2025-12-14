# Rapport de Projet : Multi-Object Visual Tracking (MLVOT)

## Introduction

Ce rapport présente l'implémentation et l'évaluation d'un système de suivi d'objets multiples (MOT) progressif, développé dans le cadre du projet MLVOT. Le projet consiste en quatre travaux pratiques (TP) qui construisent progressivement un système de tracking de plus en plus sophistiqué, allant du suivi d'un seul objet avec un filtre de Kalman à un système combinant l'apparence (ReID), la géométrie (IoU) et la prédiction de mouvement (Kalman).

## Structure du Projet

Le projet est organisé en quatre TPs progressifs :

- **TP1** : Suivi d'un seul objet avec filtre de Kalman 2D
- **TP2** : Suivi multi-objets basé sur l'Intersection over Union (IoU)
- **TP3** : Combinaison du filtre de Kalman avec l'association IoU
- **TP4** : Intégration de l'apparence (ReID) avec Kalman et IoU

## TP1 : Single Object Tracking with Kalman Filter

### Objectif et Implémentation

Le premier TP consistait à implémenter un filtre de Kalman pour le suivi d'un seul objet dans une séquence vidéo. L'objet à suivre était une balle noire détectée à l'aide d'un détecteur basé sur Canny edge detection.

### Tâches Réalisées

1. **Implémentation du filtre de Kalman** : Création de la classe `KalmanFilter` avec les trois fonctions requises :
   - `__init__()` : Initialisation avec les matrices A, B, H, Q, R selon les spécifications
   - `predict()` : Prédiction de l'état suivant et de la covariance d'erreur
   - `update()` : Mise à jour de l'état avec une nouvelle mesure

2. **Intégration avec le détecteur** : Utilisation du détecteur fourni pour extraire les centroïdes de la balle dans chaque frame.

3. **Visualisation** : Implémentation de la visualisation demandée :
   - Cercle vert pour la détection
   - Rectangle bleu pour la prédiction
   - Rectangle rouge pour l'estimation
   - Trajectoire en jaune

### Résultats

Le système a traité avec succès 93 frames de la vidéo `randomball.avi`. Le filtre de Kalman a permis de maintenir une trajectoire lisse même lors de détections manquantes, démontrant l'efficacité de la prédiction de mouvement.

### Défis Rencontrés

Le principal défi a été la compréhension de la formulation mathématique du filtre de Kalman, notamment la construction des matrices de covariance du bruit de processus (Q) et de mesure (R). La résolution a consisté à suivre rigoureusement les formules fournies dans les spécifications et à ajuster les paramètres (dt, std_acc, x_std_meas, y_std_meas) pour obtenir un suivi stable.

## TP2 : IoU-based Tracking

### Objectif et Implémentation

Le deuxième TP étendait le système au suivi de multiples objets en utilisant l'Intersection over Union (IoU) comme métrique de similarité entre les bounding boxes détectées et les tracks existants.

### Tâches Réalisées

1. **Chargement des détections** : Implémentation d'un parser pour le format MOT Challenge, permettant de charger les détections pré-générées depuis le fichier `det.txt`.

2. **Calcul de l'IoU** : Fonction pour calculer l'IoU entre deux bounding boxes, gérant les cas limites (pas d'intersection, union nulle).

3. **Algorithme hongrois** : Utilisation de `scipy.optimize.linear_sum_assignment` pour résoudre le problème d'assignation optimal entre tracks et détections.

4. **Gestion des tracks** : Implémentation de la logique de gestion des tracks :
   - Création de nouveaux tracks pour les détections non assignées
   - Mise à jour des tracks assignés
   - Suppression des tracks non matchés pendant plus de `max_age` frames

5. **Sauvegarde des résultats** : Formatage et sauvegarde des résultats au format MOT Challenge.

### Résultats

Le tracker IoU a traité 525 frames du dataset ADL-Rundle-6, générant 83 tracks distincts et 3995 détections au total. Le système a démontré sa capacité à suivre plusieurs objets simultanément, bien que certaines limitations soient apparues lors d'occlusions ou de détections manquantes.

### Défis Rencontrés

Le défi principal a été la gestion des cas où plusieurs objets se croisent ou s'occulent mutuellement. L'algorithme hongrois résout l'assignation de manière optimale, mais le choix du seuil IoU (0.3) et du `max_age` (5 frames) a nécessité plusieurs ajustements. Un seuil trop bas créait des associations erronées, tandis qu'un seuil trop élevé générait trop de nouveaux tracks.

## TP3 : Kalman-Guided IoU Tracking

### Objectif et Implémentation

Le troisième TP combinait le filtre de Kalman du TP1 avec l'association IoU du TP2. L'idée était d'utiliser la prédiction du filtre de Kalman pour améliorer l'association, notamment lors d'occlusions temporaires.

### Tâches Réalisées

1. **Adaptation du filtre de Kalman** : Modification du filtre pour travailler avec des bounding boxes plutôt que des points. Le filtre suit le centroïde de la bounding box, tandis que les dimensions (largeur, hauteur) sont conservées séparément.

2. **Prédiction avant association** : Avant de calculer l'IoU, chaque track prédit sa position future à l'aide du filtre de Kalman. L'association se fait ensuite entre les bounding boxes prédites et les détections actuelles.

3. **Mise à jour du filtre** : Après association, le filtre de Kalman est mis à jour avec la nouvelle détection, permettant une meilleure prédiction pour la frame suivante.

### Résultats

Le tracker Kalman-IoU a généré 81 tracks et 4458 détections sur les 525 frames. Comparé au TP2, on observe une légère réduction du nombre de tracks (81 vs 83), suggérant une meilleure continuité des identités. Le nombre de détections plus élevé (4458 vs 3995) indique que le système maintient les tracks plus longtemps grâce à la prédiction.

### Défis Rencontrés

L'intégration du filtre de Kalman avec les bounding boxes a nécessité de repenser la représentation de l'état. Le choix de suivre uniquement le centroïde et de conserver les dimensions séparément s'est avéré efficace. Le paramètre `max_age` a été augmenté à 10 frames pour permettre au système de récupérer après des occlusions plus longues, profitant de la prédiction du filtre de Kalman.

## TP4 : Appearance-Aware IoU-Kalman Tracker

### Objectif et Implémentation

Le quatrième TP ajoutait l'apparence des objets au système, utilisant un modèle de Re-Identification (ReID) pour extraire des features d'apparence et les combiner avec l'IoU et la prédiction Kalman.

### Tâches Réalisées

1. **Extraction de features ReID** : Implémentation d'un extracteur utilisant le modèle OSNet pré-entraîné (format ONNX). Le preprocessing inclut :
   - Redimensionnement à 64x128 pixels (format Market1501)
   - Conversion BGR vers RGB
   - Normalisation avec les statistiques ImageNet
   - Normalisation L2 des features extraites

2. **Score combiné** : Création d'un score combinant IoU et similarité d'apparence :
   ```
   S = α * IoU + β * Normalized_Similarity
   ```
   où `Normalized_Similarity = 1 / (1 + Euclidean_Distance)` et α=0.6, β=0.4.

3. **Association améliorée** : L'algorithme hongrois utilise maintenant le score combiné au lieu de l'IoU seul, permettant une meilleure distinction entre objets similaires géométriquement mais différents en apparence.

4. **Évaluation** : Implémentation d'un système d'évaluation calculant les métriques MOT standards : MOTA, IDF1, Precision, Recall, et ID_Switches.

### Résultats

Le tracker ReID-Kalman-IoU a généré 46 tracks et 4895 détections. Les métriques d'évaluation sur le dataset ADL-Rundle-6 sont les suivantes :

- **MOTA** : 0.1557 (15.57%)
- **IDF1** : 0.6011 (60.11%)
- **Precision** : 0.5803 (58.03%)
- **Recall** : 0.6235 (62.35%)
- **ID Switches** : 84
- **FPS** : 30.56

### Analyse des Résultats

Le système a traité les 525 frames en 17.18 secondes, soit environ 30.56 FPS, ce qui est acceptable pour une application en temps réel. Le nombre de tracks (46) est significativement plus bas que les TPs précédents, ce qui suggère une meilleure continuité des identités grâce à l'information d'apparence.

Cependant, le MOTA relativement bas (15.57%) indique qu'il y a encore des améliorations possibles. L'IDF1 de 60.11% est plus encourageant et montre que le système préserve bien les identités des objets. Les 84 ID switches sur 525 frames représentent environ 0.16 switch par frame, ce qui est raisonnable mais pourrait être amélioré.

### Défis Rencontrés

Le défi principal a été l'intégration du modèle ReID ONNX. L'extraction de features pour chaque détection ajoute une charge computationnelle significative. L'optimisation a consisté à :
- Extraire les features par batch plutôt qu'individuellement
- Utiliser le CPU provider d'ONNX Runtime pour la compatibilité
- Normaliser correctement les features pour une comparaison efficace

Le choix des poids α et β a également nécessité des expérimentations. Un poids trop élevé pour l'apparence (β) créait des associations erronées lorsque l'apparence changeait (éclairage, angle de vue), tandis qu'un poids trop faible ne profitait pas des avantages de l'apparence. Le compromis α=0.6, β=0.4 s'est avéré équilibré.

## Comparaison des Méthodes

### Évolution du Nombre de Tracks

- **TP2 (IoU seul)** : 83 tracks, 3995 détections
- **TP3 (Kalman + IoU)** : 81 tracks, 4458 détections
- **TP4 (ReID + Kalman + IoU)** : 46 tracks, 4895 détections

La réduction progressive du nombre de tracks, combinée à l'augmentation du nombre de détections, indique une amélioration de la continuité des identités. Le TP4, avec seulement 46 tracks, suggère que le système maintient les identités plus longtemps grâce à l'information d'apparence.

### Avantages et Limites

**TP2 (IoU)** :
- Avantages : Simple, rapide, efficace pour des objets bien séparés
- Limites : Sensible aux occlusions, peut créer de nouveaux tracks lors de détections manquantes

**TP3 (Kalman + IoU)** :
- Avantages : Meilleure robustesse aux occlusions temporaires grâce à la prédiction
- Limites : Toujours limité par la géométrie seule, difficulté à distinguer des objets similaires

**TP4 (ReID + Kalman + IoU)** :
- Avantages : Meilleure distinction entre objets, meilleure continuité des identités
- Limites : Plus lent (extraction de features), sensible aux changements d'apparence (éclairage, angle)

## Défis Globaux et Solutions

### Format des Données

Le parsing du format MOT Challenge a nécessité une attention particulière pour gérer à la fois les formats séparés par espaces et par virgules. La solution a été d'essayer les deux formats et de valider le nombre de colonnes.

### Performance

L'extraction de features ReID étant coûteuse, l'optimisation a été cruciale. L'utilisation de batch processing et de la normalisation L2 efficace a permis d'atteindre un FPS acceptable (30.56).

## Conclusion

Ce projet a permis d'implémenter et d'évaluer progressivement un système de suivi d'objets multiples, en partant d'un filtre de Kalman simple jusqu'à un système combinant apparence, géométrie et prédiction de mouvement. Chaque TP a apporté des améliorations mesurables, avec le TP4 atteignant un IDF1 de 60.11% et une précision de 58.03%.

Les principaux apprentissages incluent :
- La compréhension pratique du filtre de Kalman et de son application au tracking
- L'importance de l'algorithme hongrois pour l'association optimale
- L'apport de l'information d'apparence pour améliorer la continuité des identités
- Les compromis entre précision, vitesse et robustesse

Le système final, bien qu'ayant un MOTA modeste, démontre une bonne préservation des identités (IDF1 élevé) et une vitesse de traitement acceptable pour des applications en temps réel. Les améliorations futures pourraient inclure l'optimisation du modèle ReID, l'ajustement dynamique des poids α et β, ou l'intégration d'un détecteur plus performant.

## Annexes

### Paramètres Utilisés

**TP1** :
- dt = 0.1
- u_x = 1, u_y = 1
- std_acc = 1
- x_std_meas = 0.1, y_std_meas = 0.1

**TP2** :
- iou_threshold = 0.3
- max_age = 5
- conf_threshold = 0.7

**TP3** :
- iou_threshold = 0.3
- max_age = 10
- conf_threshold = 0.7

**TP4** :
- iou_threshold = 0.3
- max_age = 30
- conf_threshold = 0.7
- α = 0.6 (poids IoU)
- β = 0.4 (poids ReID)

### Métriques d'Évaluation (TP4)

- **MOTA (Multiple Object Tracking Accuracy)** : Mesure globale de la performance, prenant en compte les faux positifs, faux négatifs et ID switches
- **IDF1** : F1-score pour la préservation des identités
- **Precision** : Proportion de détections correctes parmi toutes les détections
- **Recall** : Proportion d'objets ground truth correctement détectés
- **ID Switches** : Nombre de changements d'identité pour un même objet

### Vidéos de Résultats

Les vidéos de visualisation des résultats sont disponibles dans les dossiers respectifs :
- `tp2/output.mp4` : Résultats du tracker IoU
- `tp3/output.mp4` : Résultats du tracker Kalman-IoU
- `tp4/output.mp4` : Résultats du tracker ReID-Kalman-IoU

Ces vidéos montrent les bounding boxes colorées pour chaque track, avec les IDs affichés, permettant de visualiser la continuité des identités et la qualité du tracking.

