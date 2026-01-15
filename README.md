# MLOps ML Project

Ce projet implémente un pipeline MLOps simple pour entraîner un modèle de classification sur le dataset Iris en utilisant scikit-learn.

## Entraînement

python scripts/train.py

## Évaluation

python scripts/evaluate.py

## Artefacts

- model.joblib
- metrics.json
- confusion_matrix.png
- report.json

## Structure du projet

- `src/`: Code source
  - `data.py`: Chargement des données
  - `features.py`: Prétraitement des features
  - `model.py`: Construction du modèle
- `scripts/`: Scripts d'exécution
  - `train.py`: Entraînement du modèle
  - `evaluate.py`: Évaluation du modèle
- `config/`: Configurations
  - `train.yaml`: Configuration d'entraînement
- `tests/`: Tests
- `artifacts/`: Artefacts générés (modèle, métriques, etc.)

## Installation

1. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. Entraîner le modèle :

   ```bash
   python scripts/train.py
   ```

2. Évaluer le modèle :

   ```bash
   python scripts/evaluate.py
   ```

Les artefacts sont sauvegardés dans le dossier `artifacts/`.

## Configuration

Modifier `config/train.yaml` pour changer les paramètres d'entraînement.
