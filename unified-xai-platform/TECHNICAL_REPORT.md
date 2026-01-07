# Rapport Technique - Unified Explainable AI Platform

**Projet de 5ème année - ESILV - Explainability AI**

**Auteurs** : Solal LEDRU, Tara MESTMAN, Tristan MOLIN & Nicolas MERLIN
**TD** : DIA TD 4

---

## Table des matières

1. [Introduction et contexte](#1-introduction-et-contexte)
2. [Analyse des repositories sources](#2-analyse-des-repositories-sources)
3. [Choix d'architecture](#3-choix-darchitecture)
4. [Implémentation des modèles](#4-implémentation-des-modèles)
5. [Implémentation des méthodes XAI](#5-implémentation-des-méthodes-xai)
6. [Pipeline de traitement des données](#6-pipeline-de-traitement-des-données)
7. [Interface utilisateur](#7-interface-utilisateur)
8. [Difficultés rencontrées et solutions](#8-difficultés-rencontrées-et-solutions)
9. [Améliorations par rapport aux repositories originaux](#9-améliorations-par-rapport-aux-repositories-originaux)
10. [Limites et perspectives](#10-limites-et-perspectives)

---

## 1. Introduction et contexte

### 1.1 Objectif du projet

L'objectif de ce projet est de fusionner deux systèmes d'intelligence artificielle explicable (XAI) en une seule plateforme unifiée :

1. **Deepfake Audio Detection** : détection d'audio synthétique vs authentique
2. **Lung Cancer Detection** : détection de tumeurs malignes sur radiographies thoraciques

La valeur ajoutée réside dans la capacité à **expliquer** les décisions des modèles via trois techniques : LIME, Grad-CAM et SHAP.

### 1.2 Pourquoi l'explicabilité est cruciale

Dans les domaines à fort enjeu (médical, sécurité), un modèle performant ne suffit pas. Il faut pouvoir :

- **Vérifier** que le modèle utilise les bonnes features (et non des artefacts du dataset)
- **Faire confiance** aux prédictions en comprenant leur logique
- **Débugger** les erreurs en identifiant ce qui a induit le modèle en erreur
- **Respecter les régulations** (RGPD Article 22 : droit à l'explication des décisions automatisées)

Le repository LungCancerDetection illustre bien ce point : les auteurs montrent des cas où Grad-CAM révèle que le modèle fait une prédiction correcte mais pour de mauvaises raisons (Figure 5 de leur README), soulignant l'importance de ne pas se fier uniquement à l'accuracy.

---

## 2. Analyse des repositories sources

### 2.1 Deepfake-Audio-Detection-with-XAI

**URL** : https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI

**Contenu disponible** :
- Modèles pré-entraînés : MobileNet, VGG16, ResNet, Custom CNN
- Application Streamlit fonctionnelle
- Notebooks Jupyter avec implémentation LIME, Grad-CAM, SHAP
- Dataset : Fake-or-Real (York University)

**Approche technique** :
Les fichiers audio sont convertis en **spectrogrammes mel** avant classification. Cette transformation est astucieuse car elle permet :
- D'utiliser des architectures CNN classiques (conçues pour les images)
- D'appliquer des techniques XAI visuelles (LIME, Grad-CAM) sur une représentation 2D
- De capturer les artefacts fréquentiels caractéristiques des deepfakes

**Performance rapportée** : ~91% d'accuracy avec MobileNet sur le dataset Fake-or-Real.

### 2.2 LungCancerDetection

**URL** : https://github.com/schaudhuri16/LungCancerDetection

**Contenu disponible** :
- README détaillé décrivant l'approche
- Pas de code exécutable ni de modèles pré-entraînés

**Approche technique** :
- Transfer learning avec AlexNet et DenseNet pré-entraînés sur ImageNet
- Fine-tuning sur le dataset CheXpert (radiographies thoraciques)
- Augmentation de données via VAE (Variational AutoEncoder)
- Grad-CAM pour l'explicabilité

**Performance rapportée** :
| Modèle | Accuracy | Recall | Precision | F1-Score |
|--------|----------|--------|-----------|----------|
| AlexNet (augmenté) | 71.48% | 75.29% | 69.31% | 72.18% |
| DenseNet (augmenté) | 73.11% | 78.89% | 70.12% | 74.24% |

### 2.3 Synthèse comparative

| Aspect | Audio Repo | Image Repo |
|--------|------------|------------|
| Code disponible | Oui (complet) | Non (README only) |
| Modèles pré-entraînés | Oui | Non |
| Interface | Streamlit | Aucune |
| Méthodes XAI | LIME, Grad-CAM, SHAP | Grad-CAM uniquement |
| État d'exécution | Fonctionnel | À implémenter |

---

## 3. Choix d'architecture

### 3.1 Framework GUI : Streamlit

Nous avons choisi **Streamlit** pour plusieurs raisons :

1. **Cohérence** : Le repository audio l'utilise déjà, facilitant l'intégration
2. **Rapidité de développement** : Création d'interfaces web sans connaissance frontend
3. **Widgets natifs** : Upload de fichiers, sliders, boutons, graphiques matplotlib
4. **Session state** : Persistance des données entre les pages (crucial pour la page de comparaison)

**Alternatives considérées** :
- **Gradio** : Plus simple mais moins flexible pour des layouts complexes
- **Flask + React** : Plus puissant mais temps de développement trop important
- **Jupyter Widgets** : Moins adapté pour une application "production-ready"

### 3.2 Architecture modulaire

Nous avons structuré le projet en modules distincts :

```
unified-xai-platform/
├── app.py                 # Point d'entrée, navigation
├── models/                # Classificateurs (séparation audio/image)
├── utils/                 # Preprocessing (séparation audio/image)
├── pages/                 # Interface utilisateur (home, comparison)
└── assets/                # Ressources (modèles sauvegardés, temp)
```

**Justification** :

| Module | Responsabilité | Avantage |
|--------|----------------|----------|
| `models/` | Chargement et inférence des modèles | Facilite l'ajout de nouveaux modèles |
| `utils/` | Prétraitement des données | Réutilisable, testable isolément |
| `pages/` | Interface et logique XAI | Séparation UI/logique métier |

### 3.3 Gestion de l'état avec session_state

Streamlit recharge le script à chaque interaction utilisateur. Pour conserver les résultats d'analyse entre les pages, nous utilisons `st.session_state` :

```python
# Sauvegarde après analyse (home.py)
st.session_state['analysis_results'] = {
    'input_type': 'audio',
    'model': 'mobilenet',
    'class_label': 0,
    'prediction': [[0.85, 0.15]],
    'image_data': spectrogram_image,
    'xai_methods': ['lime', 'gradcam']
}

# Récupération (comparison.py)
results = st.session_state['analysis_results']
```

Ce mécanisme permet à l'utilisateur de naviguer vers la page de comparaison après une analyse, sans perdre les données.

---

## 4. Implémentation des modèles

### 4.1 AudioClassifier : MobileNet pour deepfakes

**Fichier** : `models/audio/audio_classifier.py`

**Source du modèle** : Nous avons réutilisé le modèle MobileNet pré-entraîné du repository Deepfake-Audio-Detection-with-XAI, stocké dans `assets/saved_models/audio/mobilenet/`.

**Architecture** :
```
Input (224, 224, 3) → MobileNet base → Dense layers → Softmax (2 classes)
```

**Méthodes clés** :

```python
class AudioClassifier:
    def predict(self, image_array):
        """Retourne (class_label, probabilities)"""

    def predict_proba(self, image_array):
        """Retourne uniquement les probabilités (pour LIME/SHAP)"""

    def get_model(self):
        """Retourne le modèle Keras (pour Grad-CAM)"""
```

**Normalisation** : Les images sont normalisées à [0, 1] avant inférence. Le modèle a été entraîné avec cette normalisation.

### 4.2 ImageClassifier : DenseNet121 pour radiographies

**Fichier** : `models/image/image_classifier.py`

**Problème** : Le repository LungCancerDetection ne fournit pas de modèle pré-entraîné.

**Solution adoptée** : Nous avons implémenté DenseNet121 avec les poids ImageNet et une tête de classification personnalisée :

```python
def _create_densenet_model(self):
    # Base DenseNet121 sans les couches de classification
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Gel des poids

    # Nouvelle tête de classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs)
```

**Choix de DenseNet121** :
- Recommandé dans le repository original
- Dense connections : chaque couche reçoit les feature maps de toutes les couches précédentes
- Avantages : meilleur flux de gradient, réutilisation des features, moins de paramètres que VGG/ResNet

**Limitation** : Le modèle n'est pas fine-tuné sur des données médicales. Les poids ImageNet permettent d'extraire des features génériques (bords, textures) mais pas des features spécifiques aux radiographies. C'est une démonstration de l'architecture XAI, pas un outil clinique.

**Prétraitement** : Nous appliquons la normalisation DenseNet standard :
```python
x_processed = tf.keras.applications.densenet.preprocess_input(x * 255.0)
```

### 4.3 Récapitulatif des modèles

| Aspect | AudioClassifier | ImageClassifier |
|--------|-----------------|-----------------|
| Architecture | MobileNet | DenseNet121 |
| Poids | Entraînés sur Fake-or-Real | ImageNet (non fine-tuné) |
| Entrée | Spectrogramme 224×224×3 | Image 224×224×3 |
| Sortie | [P(real), P(fake)] | [P(benign), P(malignant)] |
| Stockage | Fichier local | Créé à la volée |

---

## 5. Implémentation des méthodes XAI

Les trois méthodes XAI sont implémentées directement dans `pages/home.py` et `pages/comparison.py` pour éviter les indirections inutiles.

### 5.1 LIME (Local Interpretable Model-agnostic Explanations)

**Principe** : Approximer localement le comportement d'un modèle complexe par un modèle linéaire interprétable.

**Implémentation** :

```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

def run_lime_audio(image_data, model, class_names):
    # 1. Créer l'explainer
    explainer = lime_image.LimeImageExplainer()

    # 2. Générer l'explication (1000 perturbations)
    explanation = explainer.explain_instance(
        img_array.astype('float64'),
        model.predict,
        hide_color=0,           # Couleur de masquage
        num_samples=1000        # Nombre de perturbations
    )

    # 3. Extraire le masque des features importantes
    temp, mask = explanation.get_image_and_mask(
        np.argmax(prediction[0]),
        positive_only=False,    # Afficher aussi les contributions négatives
        num_features=8,         # Top 8 features
        hide_rest=True
    )

    # 4. Visualiser avec les contours
    plt.imshow(mark_boundaries(temp, mask))
```

**Fonctionnement détaillé** :
1. L'image est segmentée en superpixels (régions de pixels similaires)
2. Pour chaque perturbation : on masque aléatoirement certains superpixels
3. On passe l'image perturbée au modèle et on récupère la prédiction
4. Un modèle linéaire (Ridge regression) est entraîné : `prédiction = Σ (wi × présence_superpixeli)`
5. Les poids `wi` indiquent l'importance de chaque superpixel

**Paramètres choisis** :
- `num_samples=1000` : Compromis entre précision et temps de calcul (~10s par explication)
- `num_features=8` : Affiche les 8 régions les plus influentes
- `hide_color=0` : Les superpixels masqués sont remplacés par du noir

### 5.2 Grad-CAM (Gradient-weighted Class Activation Mapping)

**Principe** : Utiliser les gradients de la classe prédite par rapport aux feature maps de la dernière couche convolutive pour identifier les régions importantes.

**Implémentation pour les images (DenseNet121)** :

```python
def run_gradcam_image(image_data, classifier, class_idx, class_names):
    model = classifier.get_model()
    last_conv_layer_name = classifier.get_last_conv_layer_name()  # 'conv5_block16_concat'

    # 1. Créer un modèle qui expose la sortie de la dernière couche conv
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Forward pass avec enregistrement des gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x_processed)
        class_output = predictions[:, class_idx]

    # 3. Calculer les gradients ∂class_output/∂conv_outputs
    grads = tape.gradient(class_output, conv_outputs)

    # 4. Global Average Pooling des gradients → poids par canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Combiner feature maps pondérées
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. ReLU + normalisation
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # 7. Redimensionner et appliquer colormap
    heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # 8. Superposer sur l'image originale
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
```

**Implémentation pour l'audio** :
Pour les spectrogrammes audio, nous utilisons VGG16 au lieu de MobileNet pour Grad-CAM. Ce choix vient du repository original qui utilise cette approche. VGG16 possède des couches convolutives plus facilement exploitables pour Grad-CAM.

**Couches utilisées** :
- Audio (VGG16) : `block5_conv3`
- Image (DenseNet121) : `conv5_block16_concat`

### 5.3 SHAP (SHapley Additive exPlanations)

**Principe** : Attribuer à chaque feature une valeur de Shapley, issue de la théorie des jeux coopératifs, représentant sa contribution marginale moyenne à la prédiction.

**Implémentation** :

```python
import shap
from skimage.segmentation import slic

def run_shap_audio(image_data, model, class_names):
    # 1. Segmenter l'image en superpixels
    img_uint8 = (img_array * 255).astype(np.uint8)
    segments = slic(img_uint8, n_segments=50, compactness=10, sigma=1)

    # 2. Définir la fonction de masquage
    def mask_image(mask, img, segs, background=0.0):
        masked = img.copy()
        for i, keep in enumerate(mask):
            if not keep:
                masked[segs == i] = background
        return masked

    # 3. Définir la fonction de prédiction sur les masques
    def predict_fn(masks):
        preds = []
        for mask in masks:
            masked_img = mask_image(mask, img_array, segments)
            pred = model.predict(np.expand_dims(masked_img, 0), verbose=0)
            preds.append(pred[0])
        return np.array(preds)

    # 4. Créer l'explainer SHAP (KernelExplainer pour model-agnostic)
    n_segments = len(np.unique(segments))
    background = np.ones((1, n_segments))  # Tous segments visibles
    explainer = shap.KernelExplainer(predict_fn, background)

    # 5. Calculer les valeurs SHAP
    test_mask = np.ones((1, n_segments))
    shap_values = explainer.shap_values(test_mask, nsamples=100)

    # 6. Créer la heatmap
    heatmap = np.zeros(segments.shape)
    for i, val in enumerate(values):
        heatmap[segments == i] = val
```

**Complexité** :
Le calcul exact des valeurs de Shapley est en O(2^n) où n est le nombre de features. Avec ~50 superpixels, c'est infaisable. `KernelExplainer` utilise un échantillonnage (`nsamples=100`) pour approximer les valeurs.

**Paramètres choisis** :
- `n_segments=50` : Granularité des régions (plus = plus précis mais plus lent)
- `nsamples=100` : Nombre d'échantillons pour l'approximation (plus = plus précis)
- Temps de calcul : ~30s par explication

### 5.4 Comparaison des méthodes

| Critère | LIME | Grad-CAM | SHAP |
|---------|------|----------|------|
| **Type** | Model-agnostic | Gradient-based | Model-agnostic |
| **Accès requis** | Fonction predict() | Architecture interne | Fonction predict() |
| **Temps (224×224)** | ~10s | ~1s | ~30s |
| **Granularité** | Superpixels | Continue | Superpixels |
| **Fondement théorique** | Approximation locale | Gradients CNN | Théorie des jeux |
| **Fidélité** | Locale | Globale (par classe) | Locale |

---

## 6. Pipeline de traitement des données

### 6.1 Pipeline audio

**Fichier** : `utils/audio_processing.py`

```
Fichier .wav → librosa.load() → Mel-spectrogram → Image PNG → Resize 224×224 → Normalisation [0,1]
```

**Étapes détaillées** :

```python
def create_spectrogram_from_upload(uploaded_file, temp_dir):
    # 1. Sauvegarder le fichier uploadé
    audio_path = os.path.join(temp_dir, uploaded_file.name)
    with open(audio_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # 2. Charger l'audio avec librosa
    y, sr = librosa.load(audio_path)

    # 3. Calculer le spectrogramme mel
    ms = librosa.feature.melspectrogram(y=y, sr=sr)

    # 4. Convertir en décibels (échelle logarithmique)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # 5. Afficher et sauvegarder
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig(spec_path)

    # 6. Charger et redimensionner pour le modèle
    image_data = load_img(spec_path, target_size=(224, 224))

    return image_data, spec_path
```

**Pourquoi le spectrogramme mel ?**
- **Représentation perceptuelle** : L'échelle mel est non-linéaire et correspond mieux à la perception humaine des fréquences
- **Compression** : Réduit la dimensionnalité tout en conservant l'information pertinente
- **Compatibilité CNN** : Transforme un signal 1D (audio) en image 2D analysable par des réseaux convolutifs

### 6.2 Pipeline image

**Fichier** : `utils/image_processing.py`

```
Fichier .jpg/.png → PIL.Image.open() → Convert RGB → Resize 224×224 → Normalisation [0,1] → DenseNet preprocessing
```

**Étapes détaillées** :

```python
def load_image_from_upload(uploaded_file, temp_dir, target_size=(224, 224)):
    # 1. Sauvegarder le fichier
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # 2. Charger et convertir en RGB (gère les images en niveaux de gris)
    image = Image.open(image_path).convert('RGB')

    # 3. Redimensionner
    image_resized = image.resize(target_size)

    return image_resized, image_path

def image_to_array(image_data, normalize=True):
    img_array = np.array(image_data)
    if normalize:
        img_array = img_array / 255.0
    return img_array
```

**Prétraitement DenseNet** :
```python
# Appliqué dans ImageClassifier.predict()
processed = tf.keras.applications.densenet.preprocess_input(image_array * 255.0)
```

Ce prétraitement centre les pixels selon les statistiques d'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] approximativement).

---

## 7. Interface utilisateur

### 7.1 Navigation

**Fichier** : `app.py`

```python
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Comparison"])

if page == "Home":
    render_home_page()
elif page == "Comparison":
    render_comparison_page()
```

### 7.2 Page Home

**Fichier** : `pages/home.py`

**Layout** :
```
┌─────────────────────────────────────────────────────────┐
│                    Unified XAI Platform                 │
├────────────────────────┬────────────────────────────────┤
│     Upload File        │       Configuration            │
│  [File uploader]       │  Model: [Dropdown]             │
│  [Preview audio/image] │  XAI: [Multiselect]            │
├────────────────────────┴────────────────────────────────┤
│                  [Run Analysis Button]                  │
├─────────────────────────────────────────────────────────┤
│  Results:                                               │
│  - Spectrogram / Image preview                          │
│  - Classification: REAL/FAKE ou BENIGN/MALIGNANT        │
│  - Confidence: XX%                                      │
├─────────────────────────────────────────────────────────┤
│  XAI Explanations:                                      │
│  [LIME visualization]                                   │
│  [Grad-CAM visualization]                               │
│  [SHAP visualization]                                   │
└─────────────────────────────────────────────────────────┘
```

**Détection automatique du type** :
```python
def detect_input_type(file):
    file_ext = file.name.lower().split('.')[-1]
    if file_ext in ['wav', 'mp3', 'flac', 'ogg']:
        return 'audio'
    elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
        return 'image'
    return None
```

### 7.3 Page Comparison

**Fichier** : `pages/comparison.py`

**Layout** :
```
┌─────────────────────────────────────────────────────────┐
│                    XAI Comparison                       │
├─────────────────────────────────────────────────────────┤
│  Current Analysis: Audio | MobileNet | Prediction: REAL │
├─────────────────────────────────────────────────────────┤
│  Original Input: [Spectrogram/Image]                    │
│  Classification: REAL (85.2%)                           │
├─────────────────────────────────────────────────────────┤
│  Select XAI Methods: [LIME] [Grad-CAM] [SHAP]           │
│                  [Run Comparison Button]                │
├──────────────┬──────────────┬───────────────────────────┤
│    LIME      │   Grad-CAM   │    SHAP                   │
│  [Visual]    │   [Visual]   │   [Visual]                │
│  Time: 10.2s │  Time: 1.1s  │  Time: 32.5s              │
├──────────────┴──────────────┴───────────────────────────┤
│  Summary Table:                                         │
│  | Method | Time | Speed | Type |                       │
│  | LIME   | 10s  | Medium| Model-agnostic |             │
│  | Grad-CAM| 1s  | Fast  | Gradient-based |             │
│  | SHAP   | 32s  | Slow  | Model-agnostic |             │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Difficultés rencontrées et solutions

### 8.1 Absence de modèle pré-entraîné pour les images

**Problème** : Le repository LungCancerDetection ne contient que de la documentation, pas de code ni de modèles.

**Solution** : Nous avons implémenté DenseNet121 avec les poids ImageNet. C'est une démonstration de l'architecture XAI, pas un outil clinique. Pour un usage réel, il faudrait fine-tuner sur le dataset CheXpert.

**Justification** : Le professeur a clarifié que l'objectif est de fusionner les repositories, pas nécessairement d'entraîner des modèles. Réutiliser des modèles pré-entraînés avec citation est acceptable.

### 8.2 Compatibilité Python 3.11

**Problème** : TensorFlow 2.6.2 (version du repository original) n'est pas compatible avec Python 3.11.

**Solution** : Mise à jour vers TensorFlow 2.15.0 et adaptation des autres dépendances :
```
tensorflow>=2.15.0
keras>=3.0.0
streamlit>=1.28.0
```

### 8.3 Session state perdue entre les pages

**Problème** : Les résultats d'analyse n'étaient pas disponibles sur la page de comparaison.

**Solution** : Sauvegarde explicite dans `st.session_state` après chaque analyse :
```python
st.session_state['analysis_results'] = {
    'input_type': 'audio',
    'model': 'mobilenet',
    'class_label': int(class_label),
    'prediction': prediction.tolist(),
    'image_data': image_data,
    'xai_methods': xai_methods
}
```

### 8.4 Grad-CAM pour audio

**Problème** : MobileNet ne possède pas de couche convolutive facilement exploitable pour Grad-CAM.

**Solution** : Suivant l'approche du repository original, nous utilisons VGG16 pour le Grad-CAM audio. Bien que ce ne soit pas le même modèle que pour la classification, cela permet d'avoir une visualisation des zones importantes du spectrogramme.

### 8.5 Temps de calcul SHAP

**Problème** : SHAP peut prendre plus d'une minute par image.

**Solution** : Réduction des paramètres pour la page de comparaison :
- `n_segments=30` au lieu de 50
- `nsamples=50` au lieu de 100

La page d'accueil utilise les paramètres complets pour plus de précision.

---

## 9. Améliorations par rapport aux repositories originaux

### 9.1 Interface unifiée

| Avant | Après |
|-------|-------|
| Deux repositories séparés | Une seule application |
| Audio seulement (Streamlit) | Audio + Image |
| Pas d'interface pour images | Interface complète |

### 9.2 Couverture XAI étendue

| Méthode | Repo Audio | Repo Image | Notre plateforme |
|---------|------------|------------|------------------|
| LIME | Oui | Non | Oui (audio + image) |
| Grad-CAM | Oui | Oui (doc) | Oui (audio + image) |
| SHAP | Notebooks | Non | Oui (audio + image) |

### 9.3 Page de comparaison

**Fonctionnalité absente des repos originaux** :
- Visualisation côte-à-côte de 2-3 méthodes XAI
- Métriques de temps de calcul
- Tableau comparatif des caractéristiques

### 9.4 Code modernisé

- Compatible Python 3.11+
- Dépendances à jour (TensorFlow 2.15, Streamlit 1.28)
- Architecture modulaire et maintenable

---

## 10. Limites et perspectives

### 10.1 Limites actuelles

**Modèle image non fine-tuné** :
Notre ImageClassifier utilise les poids ImageNet. Les prédictions sur des radiographies ne sont pas fiables. C'est une démonstration de l'architecture, pas un outil médical.

**Grad-CAM audio indirect** :
L'utilisation de VGG16 au lieu de MobileNet pour Grad-CAM audio est un compromis. Les visualisations montrent les zones importantes selon VGG16, pas selon le modèle de classification réel.

**Pas de support CSV** :
Le support de données tabulaires (CSV) est mentionné comme bonus dans les consignes mais n'a pas été implémenté.

### 10.2 Améliorations possibles

1. **Fine-tuning du modèle image** sur CheXpert pour des prédictions médicales réalistes
2. **Ajout d'autres modèles audio** (VGG16, ResNet, Custom CNN disponibles dans le repo original)
3. **Support CSV** avec méthodes XAI tabulaires (SHAP TreeExplainer, feature importance)
4. **Export PDF** des résultats d'analyse
5. **Caching** des modèles pour accélérer les analyses répétées
6. **Tests unitaires** pour les modules de preprocessing et les classificateurs

### 10.3 Pistes de recherche

- **Comparaison quantitative** des méthodes XAI (fidélité, stabilité, sensibilité)
- **Méthodes XAI supplémentaires** : Integrated Gradients, Layer-wise Relevance Propagation
- **Analyse multi-modale** : combiner audio et image pour une même prédiction
- **Déploiement cloud** : AWS/GCP pour démonstration en ligne

---

## Annexes

### A. Dépendances principales

```
tensorflow>=2.15.0
streamlit>=1.28.0
lime>=0.2.0
shap>=0.44.0
librosa>=0.10.0
opencv-python>=4.8.0
scikit-image>=0.21.0
matplotlib>=3.7.0
numpy>=1.24.0
Pillow>=10.0.0
```

### B. Commandes utiles

```bash
# Lancer l'application
streamlit run app.py

# Lancer avec un port spécifique
streamlit run app.py --server.port 8080

# Mode debug
streamlit run app.py --logger.level debug
```

### C. Structure des session_state

```python
st.session_state = {
    'uploaded_file': <UploadedFile>,
    'input_type': 'audio' | 'image',
    'selected_model': 'mobilenet' | 'densenet121',
    'selected_xai': ['lime', 'gradcam', 'shap'],
    'analysis_results': {
        'input_type': str,
        'model': str,
        'xai_methods': list,
        'class_label': int,
        'prediction': list,
        'image_data': PIL.Image
    }
}
```
