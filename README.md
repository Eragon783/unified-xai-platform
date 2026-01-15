# Unified Explainable AI Platform

**5th year project - ESILV - Explainability AI**

**Group**: Solal LEDRU, Tara MESTMAN, Tristan MOLIN & Nicolas MERLIN
**TD**: DIA TD 4

---

## What is this project ?

This project answer to a very concrete and actual problematic : how to make deep learning models' decisions **understandable** to human users ?

We created a platform that unifies two pre-existing decision systems :
- **Detection of deepfakes audio** : identify if a vocal recording is authetic or AI generated
- **Detection of lung cancer** : identify malignant lung cancer tumors in chest radiographs

The originality of our approach is allowing the user to **visualise why** the model is taking a specific decision, with three explainability techniques (XAI) : LIME, Grad-CAM and SHAP.

---

## Why is this project useful ?

### "Black box" issue

Deep neural networks are extremely performant, but they work like black boxes : we give them an input, they produce an output, without further explanation. This is highly problematic in critical domains :

- **Medical field** : a trained radiologist cannot trust blindly an algorithm that detects cancer without any knowledge of what elements of the X-ray were used to make the prediction
- **Security field** : when detecting deepfake audios in a security or legal context, knowing which sound artefacts betrayed the artifical audio is key

### Our solution

Our platform isn't limited to classification. It **visually** shows the zones in the image (or spectrogram) that influenced the decision. A radiologist can then verify if the model is looking in the right spots, and an audio expert can identify characteristic deepfake patterns.

---

## Functionality

| Functionality | Description |
|----------------|-------------|
| **Multi-modal upload** | Drag and drop an audio file (.wav) or image (.jpg, .png) |
| **Automated classification** | The system automatically detects the uploaded file type and apply corresponding models |
| **Three XAI methods** | LIME, Grad-CAM and SHAP are available for any entry type |
| **Side-by-side comparison & evaluation** | Visualize different XAI methods simultaneously to compare their explainations |
| **Time metrics** | Shows computation time of each method |

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (gestion of Python packages)
- Around 2 Go of free disk space (for TensorFlow dependencies)

### Installation steps

```bash
# 1. Clone or download the project
cd unified-xai-platform

# 2. Create a virtual environment (recommended)
python -m venv python_env

# 3. Activate environnement
# On Windows :
python_env\Scripts\activate
# On macOS/Linux :
source python_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

**Note** : The first execution will automatically download ImageNet weights for DenseNet121 (~30 Mo).

---

## Utilization

### Launch the app

```bash
streamlit run app.py
```

The app opens in your web browser at the following address `http://localhost:8501`.

### Typical use case

**1. Home page (Home)**

- Upload any audio file (.wav) for deepfake detection, or an X-ray image (.jpg/.png) for cancer detection
- The system automatically detects file type and only display compatible options
- Select one or more XAI methods (LIME, Grad-CAM, SHAP)
- Click on "Run Analysis"

**2. Results**

For an audio file :
- Shows generated Mel spectrogram
- Classification : "REAL" or "FAKE" with confidence %
- XAI visualization showing spectrogram areas that influenced the decision

For X-ray image :
- Shows reshaped image (224x224)
- Classification : "BENIGN" or "MALIGNANT" with confidence %
- XAI visualization highlighting suspicious areas

**3. Comparison page (Comparison)**

- Select two or three XAI methods
- Click on "Run Comparison"
- Visualize explanations side-by-side
- Compare computation time and specificities of each method

---

## Project structure

```
unified-xai-platform/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── TECHNICAL_REPORT.md         # Detailled technical report
│
├── models/
│   ├── audio/
│   │   └── audio_classifier.py # MobileNet classifier for audio
│   └── image/
│       └── image_classifier.py # DenseNet121 classifier for images
│
├── utils/
│   ├── audio_processing.py     # Audio → spectrogram conversion
│   └── image_processing.py     # Images preprocessing
│
├── pages/
│   ├── home.py                 # Principal interface + XAI
│   └── comparison.py           # XAI methods comparison
│
├── assets/
│   ├── saved_models/           # Pre-trained models
│   │   └── audio/mobilenet/    # MobileNet model for deepfakes
│   └── temp/                   # Temporary files (spectrograms, etc.)
│
└── instructions/               # Instructions for the project (reference)
```

---

## Used models

### Detection of deepfakes audio : MobileNet

**Source** : Pre-trained model from following repository [Deepfake-Audio-Detection-with-XAI](https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI)

**Working principle** :
1. The audio is converted to Mel spectrogram (visual representation of frequencies in time)
2. The spectrogram is reshaped in a 224x224 pixels image
3. MobileNet, initially created for image classification, analyse this spectrogram
4. Output : "real" vs "fake" probability

**Why it works** : Audio deepfakes audio leave a characteristic trace in the frequency domain. The spectrogram captures these artefacts as visual patterns that the neural network can learn to spot.

### Detection of lung cancer : DenseNet121

**Source** : We implemented DenseNet121 with pre-trained weights on ImageNet, following the approach described in the following repository [LungCancerDetection](https://github.com/schaudhuri16/LungCancerDetection).

**Why DenseNet** : This network uses "dense connections" where each layer receives the feature maps of all previous layers. This favors reausability of features and improve gradiant flow. This is particularly useful for medical images where very subtle details matter.

**Important limitations** : Our model uses ImageNet weights (natural images) and is not fine-tuned on real medical data. This project is to showcase XAI methods, not a medical diagnosis tool.

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
