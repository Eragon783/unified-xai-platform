# Unified Explainable AI Platform

**Projet de 5ème année - ESILV - Explainability AI**

**Groupe**: Solal LEDRU, Tara MESTMAN, Tristan MOLIN & Nicolas MERLIN
**TD**: DIA TD 4

---

## Qu'est-ce que ce projet ?

Ce projet répond à une problématique concrète : comment rendre les décisions des modèles de deep learning **compréhensibles** pour un utilisateur humain ?

Nous avons créé une plateforme qui unifie deux systèmes de détection existants :
- **Détection de deepfakes audio** : identifier si un enregistrement vocal est authentique ou généré artificiellement
- **Détection de cancer pulmonaire** : analyser des radiographies thoraciques pour repérer des tumeurs malignes

L'originalité de notre approche est de permettre à l'utilisateur de **visualiser pourquoi** le modèle prend sa décision, grâce à trois techniques d'explicabilité (XAI) : LIME, Grad-CAM et SHAP.

---

## Pourquoi ce projet est-il pertinent ?

### Le problème des "boîtes noires"

Les réseaux de neurones profonds sont extrêmement performants, mais ils fonctionnent comme des boîtes noires : on leur donne une entrée, ils produisent une sortie, sans explication. C'est problématique dans des domaines critiques :

- **En médecine** : un radiologue ne peut pas faire confiance aveuglément à un algorithme qui lui dit "cancer" sans savoir sur quels éléments de l'image il se base
- **En sécurité** : détecter un deepfake audio nécessite de comprendre quels artefacts sonores ont trahi la synthèse

### Notre réponse

Notre plateforme ne se contente pas de classifier. Elle montre **visuellement** les zones de l'image (ou du spectrogramme) qui ont influencé la décision. Un radiologue peut ainsi vérifier si le modèle regarde au bon endroit, et un expert en audio peut identifier les patterns caractéristiques des deepfakes.

---

## Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| **Upload multi-modal** | Glisser-déposer un fichier audio (.wav) ou une image (.jpg, .png) |
| **Classification automatique** | Le système détecte le type de fichier et applique le modèle approprié |
| **Trois méthodes XAI** | LIME, Grad-CAM et SHAP disponibles pour chaque type d'entrée |
| **Comparaison côte-à-côte** | Visualiser plusieurs méthodes XAI simultanément pour comparer leurs explications |
| **Métriques de temps** | Affichage du temps de calcul de chaque méthode |

---

## Installation

### Prérequis

- Python 3.11 ou supérieur
- pip (gestionnaire de packages Python)
- Environ 2 Go d'espace disque (pour les dépendances TensorFlow)

### Étapes d'installation

```bash
# 1. Cloner ou télécharger le projet
cd unified-xai-platform

# 2. Créer un environnement virtuel (recommandé)
python -m venv python_env

# 3. Activer l'environnement
# Sur Windows :
python_env\Scripts\activate
# Sur macOS/Linux :
source python_env/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt
```

**Note** : La première exécution téléchargera automatiquement les poids ImageNet pour DenseNet121 (~30 Mo).

---

## Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre dans votre navigateur à l'adresse `http://localhost:8501`.

### Scénario d'utilisation typique

**1. Page d'accueil (Home)**

- Uploadez un fichier audio (.wav) pour la détection de deepfake, ou une image de radiographie (.jpg/.png) pour la détection de cancer
- Le système détecte automatiquement le type et affiche les options correspondantes
- Sélectionnez une ou plusieurs méthodes XAI (LIME, Grad-CAM, SHAP)
- Cliquez sur "Run Analysis"

**2. Résultats**

Pour un fichier audio :
- Affichage du spectrogramme mel généré
- Classification : "REAL" ou "FAKE" avec le pourcentage de confiance
- Visualisations XAI montrant les zones du spectrogramme qui ont influencé la décision

Pour une image médicale :
- Affichage de l'image redimensionnée (224x224)
- Classification : "BENIGN" ou "MALIGNANT" avec le pourcentage de confiance
- Visualisations XAI mettant en évidence les régions suspectes

**3. Page de comparaison (Comparison)**

- Sélectionnez 2 ou 3 méthodes XAI
- Cliquez sur "Run Comparison"
- Visualisez les explications côte-à-côte
- Comparez les temps de calcul et les caractéristiques de chaque méthode

---

## Structure du projet

```
unified-xai-platform/
├── app.py                      # Point d'entrée Streamlit
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
├── TECHNICAL_REPORT.md         # Rapport technique détaillé
│
├── models/
│   ├── audio/
│   │   └── audio_classifier.py # Classificateur MobileNet pour audio
│   └── image/
│       └── image_classifier.py # Classificateur DenseNet121 pour images
│
├── utils/
│   ├── audio_processing.py     # Conversion audio → spectrogramme
│   └── image_processing.py     # Prétraitement des images
│
├── pages/
│   ├── home.py                 # Interface principale + XAI
│   └── comparison.py           # Comparaison des méthodes XAI
│
├── assets/
│   ├── saved_models/           # Modèles pré-entraînés
│   │   └── audio/mobilenet/    # Modèle MobileNet pour deepfakes
│   └── temp/                   # Fichiers temporaires (spectrogrammes, etc.)
│
└── instructions/               # Consignes du projet (référence)
```

---

## Les modèles utilisés

### Détection de deepfakes audio : MobileNet

**Source** : Modèle pré-entraîné du repository [Deepfake-Audio-Detection-with-XAI](https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI)

**Fonctionnement** :
1. L'audio est converti en spectrogramme mel (représentation visuelle des fréquences dans le temps)
2. Le spectrogramme est redimensionné en 224x224 pixels
3. MobileNet, initialement conçu pour la classification d'images, analyse ce spectrogramme
4. Sortie : probabilité "real" vs "fake"

**Pourquoi ça marche** : Les deepfakes audio laissent des traces caractéristiques dans le domaine fréquentiel. Le spectrogramme capture ces artefacts sous forme de patterns visuels que le réseau peut apprendre à reconnaître.

### Détection de cancer pulmonaire : DenseNet121

**Source** : Nous avons implémenté DenseNet121 avec les poids pré-entraînés sur ImageNet, en suivant l'approche décrite dans le repository [LungCancerDetection](https://github.com/schaudhuri16/LungCancerDetection).

**Pourquoi DenseNet** : Ce réseau utilise des "dense connections" où chaque couche reçoit les feature maps de toutes les couches précédentes. Cela favorise la réutilisation des features et améliore le flux de gradient, ce qui est particulièrement utile pour les images médicales où les détails subtils comptent.

**Limitation importante** : Notre modèle utilise des poids ImageNet (images naturelles) et n'a pas été fine-tuné sur des données médicales réelles. Il s'agit d'une démonstration des techniques XAI, pas d'un outil de diagnostic médical.

---

## Les méthodes d'explicabilité (XAI)

### LIME (Local Interpretable Model-agnostic Explanations)

**Principe** : LIME traite le modèle comme une boîte noire et l'interroge de manière répétée.

**Comment ça fonctionne** :
1. L'image est segmentée en "superpixels" (régions cohérentes)
2. On génère des milliers de versions perturbées en masquant certains superpixels
3. On observe comment la prédiction du modèle change
4. Un modèle linéaire simple est entraîné pour approximer localement le comportement du réseau
5. Les superpixels avec les plus forts coefficients sont mis en évidence

**Avantage** : Fonctionne avec n'importe quel modèle, sans accès à son architecture interne.

**Inconvénient** : Relativement lent (1000 prédictions par explication).

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**Principe** : Grad-CAM exploite les gradients du réseau pour identifier les régions importantes.

**Comment ça fonctionne** :
1. On effectue une passe avant (forward pass) jusqu'à la dernière couche convolutive
2. On calcule le gradient de la classe prédite par rapport aux feature maps
3. Ces gradients sont moyennés pour obtenir un poids par canal
4. Les feature maps sont combinées selon ces poids
5. Une fonction ReLU garde uniquement les contributions positives

**Avantage** : Très rapide (une seule passe arrière), résolution fidèle aux features apprises.

**Inconvénient** : Nécessite l'accès aux couches internes du réseau (spécifique aux CNN).

### SHAP (SHapley Additive exPlanations)

**Principe** : SHAP s'appuie sur la théorie des jeux pour attribuer une "contribution" à chaque feature.

**Comment ça fonctionne** :
1. L'image est segmentée en superpixels
2. Pour chaque segment, on calcule sa valeur de Shapley : la contribution marginale moyenne de ce segment à la prédiction, sur toutes les combinaisons possibles de segments
3. Ces valeurs sont visualisées sous forme de heatmap

**Avantage** : Fondement théorique solide, attributions qui somment exactement à la différence entre la prédiction et la baseline.

**Inconvénient** : Très coûteux en calcul (exponentiel en nombre de features, d'où l'approximation par échantillonnage).

---

## Fichiers de test

Pour tester l'application, vous pouvez utiliser :

**Audio** : N'importe quel fichier .wav de quelques secondes. Des exemples sont disponibles dans le dataset [Fake-or-Real](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset).

**Images** : Des radiographies thoraciques au format .jpg ou .png. Des exemples sont disponibles dans le dataset [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert).

---

## Déclaration d'utilisation d'IA générative

Conformément aux exigences du projet, nous déclarons l'utilisation des outils suivants :

### Outils utilisés

| Outil | Modèle | Version |
|-------|--------|---------|
| Claude Code (Anthropic) | Claude | CLI Extension VSCode |

### Utilisation détaillée

**Développement de code** :
- Structuration de l'architecture du projet (séparation models/, utils/, pages/)
- Implémentation de l'interface Streamlit
- Intégration des bibliothèques XAI (LIME, SHAP, tf-keras-vis)
- Création des pipelines de prétraitement audio et image
- Débogage des problèmes de compatibilité Python 3.11

**Documentation** :
- Rédaction du README.md
- Rédaction du rapport technique (TECHNICAL_REPORT.md)
- Commentaires dans le code

### Contributions humaines

- Définition des exigences et prise de décisions techniques
- Tests manuels de l'application
- Validation des résultats XAI
- Revue et correction du code et de la documentation