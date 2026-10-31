````markdown
# MedSAM - Segmentation Médicale Interactive et Comparative

Ce projet est une plateforme web interactive développée avec Flask. Elle permet d'importer des images médicales, de dessiner une zone d'intérêt (bounding box), et de comparer les résultats de segmentation entre des méthodes conventionnelles (Seuillage d'Otsu, Ligne de partage des eaux) et un modèle de Deep Learning de pointe spécialisé dans le médical (MedSAM).

## Fonctionnalités

- **Importation d'images** : Upload d'images médicales (IRM, Scanner, etc.) aux formats PNG, JPG, JPEG, BMP, TIF.
- **Interface interactive** : Outil de dessin (Canvas HTML5/JS) permettant à l'utilisateur d'entourer la zone à segmenter.
- **Traitement en parallèle** :
  - Application des méthodes conventionnelles (Otsu, Watershed).
  - Inférence du modèle MedSAM (Vision Transformer) guidée par la boîte englobante de l'utilisateur.
- **Visualisation comparative** : Affichage côte à côte des 5 résultats (Image originale, Otsu, Watershed, Masque MedSAM, Superposition MedSAM).

## Lancement rapide avec Docker (Recommandé)

La méthode la plus simple pour exécuter ce projet, sans se soucier des dépendances Python ou du système d'exploitation, est d'utiliser Docker.

**Prérequis** : Avoir [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et en cours d'exécution.

1. Ouvrez un terminal dans le dossier du projet.
2. Construisez l'image Docker (cela peut prendre quelques minutes) :
   ```bash
   docker build -t medsam-app .
   ```
````

3. Lancez le conteneur :
   ```bash
   docker run -p 5001:5001 medsam-app
   ```
4. Ouvrez votre navigateur à l'adresse : **http://localhost:5001**

_Note : Lors du premier lancement, l'application télécharge automatiquement le modèle MedSAM (~375 Mo) depuis Hugging Face. L'inférence sur CPU peut prendre entre 10 et 30 secondes par image._

## Lancement local (Sans Docker)

Si vous préférez exécuter le projet directement sur votre machine avec Python :

1. **Créer et activer un environnement virtuel** :

   ```bash
   python -m venv venv
   # Sur Windows :
   .\venv\Scripts\activate
   # Sur Mac/Linux :
   source venv/bin/activate
   ```

2. **Installer les dépendances** :

   ```bash

   pip install -r requirements.txt
   ```

3. **Lancer l'application** :

   ```bash
   python app.py
   ```

4. **Accéder à l'application** : Ouvrez votre navigateur à l'adresse **http://127.0.0.1:5001**.

## Guide d'utilisation

1. **Importer** : Sur la page d'accueil, sélectionnez une image médicale et cliquez sur "Continuer vers le dessin".
2. **Dessiner** : Sur la page suivante, cliquez et faites glisser la souris pour tracer un rectangle (bounding box) autour de la structure d'intérêt (par exemple, le cartilage ou l'os).
3. **Analyser** : Cliquez sur le bouton "Segmenter avec MedSAM". Le système exécute alors en parallèle les algorithmes conventionnels et le réseau de neurones profond.
4. **Comparer** : Une fois l'analyse terminée, la page de résultats s'affiche et présente les différentes approches côte à côte pour évaluation visuelle.

## Architecture du projet

- `app.py` : Point d'entrée de l'application Flask. Gère le routage, l'upload, l'inférence MedSAM (via `transformers`) et l'exécution des méthodes conventionnelles.
- `config.py` : Configuration centralisée (chemins des dossiers, identifiant du modèle Hugging Face).
- `modules/` : Contient les scripts de traitement d'image conventionnel.
  - `preprocessing.py` : Prétraitement (CLAHE, débruitage).
  - `conventional_seg.py` : Algorithmes de segmentation (Otsu, Watershed, etc.).
- `templates/` : Fichiers HTML (Jinja2).
  - `index.html` : Page d'importation.
  - `draw.html` : Interface de dessin de la bounding box (Canvas).
  - `results.html` : Grille de comparaison des résultats.
- `static/` : Fichiers statiques (CSS, images uploadées, images générées).
- `Dockerfile` : Recette de construction de l'image Docker.

## À propos du modèle de Deep Learning

Le modèle utilisé est **MedSAM** (_Segment Anything in Medical Images_), publié dans la revue _Nature Communications_ en 2024. Contrairement aux modèles classiques, MedSAM a été spécifiquement ré-entraîné sur plus de 1,5 million de paires image/masque médicales. Il s'agit d'un modèle "promptable" : il nécessite que l'utilisateur lui indique la zone d'intérêt (via un rectangle) pour fonctionner de manière optimale.

```

```
