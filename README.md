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

## Explanability methods (XAI)

### LIME (Local Interpretable Model-agnostic Explanations)

**Principle** : LIME treats the model like a black box and queries it repeatedly.

**How it works** :
1. The image is segmented in "superpixels" (coherent regions)
2. We generate thousands of pertubated versions while hiding some superpixels
3. We observe how the model prediction changes
4. A simple linear model is trained to locally approximate the behavior of the network
5. Superpixels with strongest coefficients are highlighted

**Avantages** : Works with any model without needing access to its internal architecture.

**Disadvantages** : Relatively slow (1000 predictions by explanation).

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**Principle** : Grad-CAM exploits the network's gradiants to identify important regions.

**How it works** :
1. We do a forward pass up to the last convolutional layer
2. We compute the gradiant of the predicted class compared to the feature maps
3. These gradiants are averaged to obtain one weight per canal
4. Features maps are combined following these weights
5. A ReLU function keeps only positive contributions

**Avantages** : Veru fast (only one forward pass), resolution faithful to learned features.

**Disadvantages** : Needs access to internal layer of the network (specific for CNN).

### SHAP (SHapley Additive exPlanations)

**Principle** : SHAP uses game theory to give one "contribution" to each feature.

**How it works** :
1. The image is segmented in "superpixels"
2. For each segment, we compute its Shapley value : the mean marginal contribution of this segment to the prediction, on every possible segment combinations
3. Values are shown in a heatmap

**Avantages** : Solid theorical fundation, attributes that sum up exactly to the difference between the prediction and the baseline.

**Disadvantages** : High computational cost (exponential in number of features, hence the approximation by sampling).

---

## Test files

To test the app you can use :

**Audio** : Any .wav file of a few seconds long. Exemples are available in this dataset [Fake-or-Real](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset).

**Images** : Chest X-rays in .jpg or .png format. Exemples are available in this dataset [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert).

---

## Declaration of GenAI usage
Like asked by the project's requirements, we declare the usage of the following tools :

### Used tools

| Outil | Model | Version |
|-------|--------|---------|
| Claude Code (Anthropic) | Claude | CLI Extension VSCode |

### Detailed usage

**Code development** :
- Project architecture structuration (split of models/, utils/, pages/)
- Streamlit interface implementation
- Integration of XAI libraries (LIME, SHAP, tf-keras-vis)
- Audio and image preprocessing pipelines creation
- Debug of compatibility issues with Python 3.11

**Documentation** :
- Redaction of README.md (this file)
- Rdaction of technical report (TECHNICAL_REPORT.md)
- Code commentaries

### Human contribution

- Requirements definition and technical decisions making
- Manual tests of the app
- Validation of XAI results
- Review and corrections of the code and documentation
