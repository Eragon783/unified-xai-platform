# Technical Report - Unified Explainable AI Platform

**5th Year Project - ESILV - Explainability AI**

**Authors**: Solal LEDRU, Tara MESTMAN, Tristan MOLIN & Nicolas MERLIN  
**Class**: DIA TD 4

---

## Table of Contents

1. [Introduction and Context](#1-introduction-and-context)
2. [Source Repositories Analysis](#2-source-repositories-analysis)
3. [Architecture Choices](#3-architecture-choices)
4. [Models Implementation](#4-models-implementation)
5. [XAI Methods Implementation](#5-xai-methods-implementation)
6. [Data Processing Pipeline](#6-data-processing-pipeline)
7. [User Interface](#7-user-interface)
8. [Challenges Encountered and Solutions](#8-challenges-encountered-and-solutions)
9. [Improvements Over Original Repositories](#9-improvements-over-original-repositories)
10. [Limitations and Perspectives](#10-limitations-and-perspectives)

---

## 1. Introduction and Context

### 1.1 Project Objective

The objective of this project is to merge several explainable artificial intelligence (XAI) systems into a single unified platform:

1. **Deepfake Audio Detection**: detection of synthetic vs authentic audio
2. **Lung Cancer Detection**: detection of malignant tumors on chest X-rays
3. **Fraud Detection**: detection of fraudulent transactions from tabular data (CSV)

The added value lies in the ability to **explain** model decisions through techniques adapted to each data type: LIME, Grad-CAM and SHAP for images/audio, and LIME Tabular, SHAP TreeExplainer and Feature Importance for tabular data.

### 1.2 Why Explainability is Crucial

In high-stakes domains (medical, security, finance), a high-performing model is not enough. We need to be able to:

- **Verify** that the model uses the right features (and not dataset artifacts)
- **Trust** predictions by understanding their logic
- **Debug** errors by identifying what misled the model
- **Comply with regulations** (GDPR Article 22: right to explanation of automated decisions)
- **Justify decisions** to clients (e.g., why a transaction was blocked)

The LungCancerDetection repository illustrates this well: the authors show cases where Grad-CAM reveals that the model makes a correct prediction but for the wrong reasons (Figure 5 of their README), highlighting the importance of not relying solely on accuracy.

---

## 2. Source Repositories Analysis

### 2.1 Deepfake-Audio-Detection-with-XAI

**URL**: https://github.com/Guri10/Deepfake-Audio-Detection-with-XAI

**Available content**:
- Pre-trained models: MobileNet, VGG16, ResNet, Custom CNN
- Functional Streamlit application
- Jupyter notebooks with LIME, Grad-CAM, SHAP implementation
- Dataset: Fake-or-Real (York University)

**Technical approach**:
Audio files are converted to **mel spectrograms** before classification. This transformation is clever because it allows:
- Using classic CNN architectures (designed for images)
- Applying visual XAI techniques (LIME, Grad-CAM) on a 2D representation
- Capturing frequency artifacts characteristic of deepfakes

**Reported performance**: ~91% accuracy with MobileNet on the Fake-or-Real dataset.

### 2.2 LungCancerDetection

**URL**: https://github.com/schaudhuri16/LungCancerDetection

**Available content**:
- Detailed README describing the approach
- No executable code or pre-trained models

**Technical approach**:
- Transfer learning with AlexNet and DenseNet pre-trained on ImageNet
- Fine-tuning on the CheXpert dataset (chest X-rays)
- Data augmentation via VAE (Variational AutoEncoder)
- Grad-CAM for explainability

**Reported performance**:
| Model | Accuracy | Recall | Precision | F1-Score |
|-------|----------|--------|-----------|----------|
| AlexNet (augmented) | 71.48% | 75.29% | 69.31% | 72.18% |
| DenseNet (augmented) | 73.11% | 78.89% | 70.12% | 74.24% |

### 2.3 Comparative Summary

| Aspect | Audio Repo | Image Repo |
|--------|------------|------------|
| Code available | Yes (complete) | No (README only) |
| Pre-trained models | Yes | No |
| Interface | Streamlit | None |
| XAI methods | LIME, Grad-CAM, SHAP | Grad-CAM only |
| Execution state | Functional | To implement |

---

## 3. Architecture Choices

### 3.1 GUI Framework: Streamlit

We chose **Streamlit** for several reasons:

1. **Consistency**: The audio repository already uses it, facilitating integration
2. **Development speed**: Creating web interfaces without frontend knowledge
3. **Native widgets**: File upload, sliders, buttons, matplotlib charts
4. **Session state**: Data persistence between pages (crucial for the comparison page)

**Alternatives considered**:
- **Gradio**: Simpler but less flexible for complex layouts
- **Flask + React**: More powerful but development time too long
- **Jupyter Widgets**: Less suitable for a "production-ready" application

### 3.2 Modular Architecture

We structured the project into distinct modules:

```
unified-xai-platform/
├── app.py                 # Entry point, navigation
├── models/                # Classifiers (audio/image/tabular)
├── utils/                 # Preprocessing (audio/image/tabular)
├── views/                 # User interface (home, comparison)
├── assets/                # Resources (saved models, temp)
└── file_test/             # Test files (CSV fraud)
```

**Justification**:

| Module | Responsibility | Advantage |
|--------|----------------|-----------|
| `models/` | Model loading and inference | Facilitates adding new models |
| `utils/` | Data preprocessing | Reusable, testable in isolation |
| `views/` | Interface and XAI logic | UI/business logic separation |

### 3.3 State Management with session_state

Streamlit reloads the script on each user interaction. To keep analysis results between pages, we use `st.session_state`:

```python
# Save after analysis (home.py)
st.session_state['analysis_results'] = {
    'input_type': 'audio',
    'model': 'mobilenet',
    'class_label': 0,
    'prediction': [[0.85, 0.15]],
    'image_data': spectrogram_image,
    'xai_methods': ['lime', 'gradcam']
}

# Retrieval (comparison.py)
results = st.session_state['analysis_results']
```

This mechanism allows the user to navigate to the comparison page after an analysis without losing data.

---

## 4. Models Implementation

### 4.1 AudioClassifier: MobileNet for deepfakes

**File**: `models/audio/audio_classifier.py`

**Model source**: We reused the pre-trained MobileNet model from the Deepfake-Audio-Detection-with-XAI repository, stored in `assets/saved_models/audio/mobilenet/`.

**Architecture**:
```
Input (224, 224, 3) → MobileNet base → Dense layers → Softmax (2 classes)
```

**Key methods**:

```python
class AudioClassifier:
    def predict(self, image_array):
        """Returns (class_label, probabilities)"""

    def predict_proba(self, image_array):
        """Returns only probabilities (for LIME/SHAP)"""

    def get_model(self):
        """Returns the Keras model (for Grad-CAM)"""
```

**Normalization**: Images are normalized to [0, 1] before inference. The model was trained with this normalization.

### 4.2 ImageClassifier: DenseNet121 for X-rays

**File**: `models/image/image_classifier.py`

**Problem**: The LungCancerDetection repository does not provide a pre-trained model.

**Adopted solution**: We implemented DenseNet121 with ImageNet weights and a custom classification head:

```python
def _create_densenet_model(self):
    # Base DenseNet121 without classification layers
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze weights

    # New classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs)
```

**DenseNet121 choice**:
- Recommended in the original repository
- Dense connections: each layer receives feature maps from all previous layers
- Advantages: better gradient flow, feature reuse, fewer parameters than VGG/ResNet

**Limitation**: The model is not fine-tuned on medical data. ImageNet weights allow extracting generic features (edges, textures) but not X-ray-specific features. This is a demonstration of XAI architecture, not a clinical tool.

**Preprocessing**: We apply standard DenseNet normalization:
```python
x_processed = tf.keras.applications.densenet.preprocess_input(x * 255.0)
```

### 4.3 FraudClassifier: RandomForest for tabular data

**File**: `models/tabular/fraud_classifier.py`

**Model source**: RandomForest model trained on a format similar to the Kaggle Credit Card Fraud Detection dataset.

**Architecture**:
```
Input (29 features: V1-V28 + Amount) → RandomForest (100 trees) → Softmax (2 classes)
```

**Key methods**:

```python
class FraudClassifier:
    def predict(self, features_array):
        """Returns (class_label, probabilities)"""

    def predict_proba(self, features_array):
        """Returns only probabilities (for LIME/SHAP)"""

    def get_model(self):
        """Returns the sklearn model (for SHAP TreeExplainer)"""

    def get_feature_importance(self):
        """Returns RandomForest feature importance"""
```

**CSV input format**:
The CSV file must contain the columns:
- `V1` to `V28`: Features from PCA (anonymized)
- `Amount`: Transaction amount
- `Class` (optional): Real label (0=legitimate, 1=fraud)

### 4.4 Models Summary

| Aspect | AudioClassifier | ImageClassifier | FraudClassifier |
|--------|-----------------|-----------------|-----------------|
| Architecture | MobileNet | DenseNet121 | RandomForest |
| Weights | Trained on Fake-or-Real | ImageNet (not fine-tuned) | Synthetic data |
| Input | Spectrogram 224x224x3 | Image 224x224x3 | 29 features vector |
| Output | [P(real), P(fake)] | [P(benign), P(malignant)] | [P(legitimate), P(fraud)] |
| Storage | Local file | Created on-the-fly | Created on-the-fly |

---

## 5. XAI Methods Implementation

XAI methods are implemented directly in `views/home.py` and `views/comparison.py` to avoid unnecessary indirections.

### 5.1 LIME (Local Interpretable Model-agnostic Explanations)

**Principle**: Locally approximate the behavior of a complex model with an interpretable linear model.

**Implementation**:

```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

def run_lime_audio(image_data, model, class_names):
    # 1. Create the explainer
    explainer = lime_image.LimeImageExplainer()

    # 2. Generate explanation (1000 perturbations)
    explanation = explainer.explain_instance(
        img_array.astype('float64'),
        model.predict,
        hide_color=0,           # Masking color
        num_samples=1000        # Number of perturbations
    )

    # 3. Extract important features mask
    temp, mask = explanation.get_image_and_mask(
        np.argmax(prediction[0]),
        positive_only=False,    # Also show negative contributions
        num_features=8,         # Top 8 features
        hide_rest=True
    )

    # 4. Visualize with boundaries
    plt.imshow(mark_boundaries(temp, mask))
```

**Detailed operation**:
1. The image is segmented into superpixels (regions of similar pixels)
2. For each perturbation: some superpixels are randomly masked
3. The perturbed image is passed to the model and the prediction is retrieved
4. A linear model (Ridge regression) is trained: `prediction = Sum(wi x presence_superpixeli)`
5. The weights `wi` indicate the importance of each superpixel

**Chosen parameters**:
- `num_samples=1000`: Compromise between precision and computation time (~10s per explanation)
- `num_features=8`: Displays the 8 most influential regions
- `hide_color=0`: Masked superpixels are replaced with black

### 5.2 Grad-CAM (Gradient-weighted Class Activation Mapping)

**Principle**: Use the gradients of the predicted class with respect to the feature maps of the last convolutional layer to identify important regions.

**Implementation for images (DenseNet121)**:

```python
def run_gradcam_image(image_data, classifier, class_idx, class_names):
    model = classifier.get_model()
    last_conv_layer_name = classifier.get_last_conv_layer_name()  # 'conv5_block16_concat'

    # 1. Create a model that exposes the output of the last conv layer
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Forward pass with gradient recording
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x_processed)
        class_output = predictions[:, class_idx]

    # 3. Calculate gradients dclass_output/dconv_outputs
    grads = tape.gradient(class_output, conv_outputs)

    # 4. Global Average Pooling of gradients -> weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Combine weighted feature maps
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. ReLU + normalization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # 7. Resize and apply colormap
    heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # 8. Superimpose on original image
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
```

**Implementation for audio**:
For audio spectrograms, we use VGG16 instead of MobileNet for Grad-CAM. This choice comes from the original repository which uses this approach. VGG16 has convolutional layers more easily exploitable for Grad-CAM.

**Layers used**:
- Audio (VGG16): `block5_conv3`
- Image (DenseNet121): `conv5_block16_concat`

### 5.3 SHAP (SHapley Additive exPlanations)

**Principle**: Assign to each feature a Shapley value, from cooperative game theory, representing its average marginal contribution to the prediction.

**Implementation**:

```python
import shap
from skimage.segmentation import slic

def run_shap_audio(image_data, model, class_names):
    # 1. Segment image into superpixels
    img_uint8 = (img_array * 255).astype(np.uint8)
    segments = slic(img_uint8, n_segments=50, compactness=10, sigma=1)

    # 2. Define masking function
    def mask_image(mask, img, segs, background=0.0):
        masked = img.copy()
        for i, keep in enumerate(mask):
            if not keep:
                masked[segs == i] = background
        return masked

    # 3. Define prediction function on masks
    def predict_fn(masks):
        preds = []
        for mask in masks:
            masked_img = mask_image(mask, img_array, segments)
            pred = model.predict(np.expand_dims(masked_img, 0), verbose=0)
            preds.append(pred[0])
        return np.array(preds)

    # 4. Create SHAP explainer (KernelExplainer for model-agnostic)
    n_segments = len(np.unique(segments))
    background = np.ones((1, n_segments))  # All segments visible
    explainer = shap.KernelExplainer(predict_fn, background)

    # 5. Calculate SHAP values
    test_mask = np.ones((1, n_segments))
    shap_values = explainer.shap_values(test_mask, nsamples=100)

    # 6. Create heatmap
    heatmap = np.zeros(segments.shape)
    for i, val in enumerate(values):
        heatmap[segments == i] = val
```

**Complexity**:
Exact calculation of Shapley values is O(2^n) where n is the number of features. With ~50 superpixels, this is infeasible. `KernelExplainer` uses sampling (`nsamples=100`) to approximate values.

**Chosen parameters**:
- `n_segments=50`: Region granularity (more = more precise but slower)
- `nsamples=100`: Number of samples for approximation (more = more precise)
- Computation time: ~30s per explanation

### 5.4 XAI Methods for Tabular Data

For tabular data (fraud detection), we use specific XAI methods:

#### 5.4.1 LIME Tabular

**Implementation**:

```python
from lime.lime_tabular import LimeTabularExplainer

def run_lime_tabular(data_row, classifier, feature_names):
    # 1. Generate synthetic training data for LIME
    training_data = generate_synthetic_data(500)

    # 2. Create explainer
    explainer = LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=['Legitimate', 'Fraud'],
        mode='classification'
    )

    # 3. Generate explanation
    explanation = explainer.explain_instance(
        data_row,
        classifier.predict_proba,
        num_features=10
    )

    # 4. Visualize as bar chart
    fig = explanation.as_pyplot_figure()
```

**Important note**: LIME Tabular requires a reference dataset to perturb values. We generate 500 synthetic samples for this.

#### 5.4.2 SHAP TreeExplainer

**Implementation**:

```python
import shap

def run_shap_tabular(data_row, classifier, feature_names):
    model = classifier.get_model()

    # TreeExplainer optimized for RandomForest
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values (exact, no approximation)
    shap_values = explainer.shap_values(data_row.reshape(1, -1))

    # Visualization
    shap.summary_plot(shap_values, feature_names=feature_names)
```

**Advantage**: `TreeExplainer` calculates Shapley values exactly for tree-based models, without approximation.

#### 5.4.3 Feature Importance

**Implementation**:

```python
def run_feature_importance(classifier, feature_names):
    # Get native RandomForest importance
    importances = classifier.get_feature_importance()

    # Sort by decreasing importance
    indices = np.argsort(importances)[::-1][:10]

    # Visualize
    plt.barh(feature_names[indices], importances[indices])
```

**Foundation**: Importance is calculated by RandomForest as the average impurity reduction (Gini) provided by each feature across all trees.

### 5.5 Methods Comparison

#### For images/audio:

| Criterion | LIME | Grad-CAM | SHAP |
|-----------|------|----------|------|
| **Type** | Model-agnostic | Gradient-based | Model-agnostic |
| **Required access** | predict() function | Internal architecture | predict() function |
| **Time (224x224)** | ~10s | ~1s | ~30s |
| **Granularity** | Superpixels | Continuous | Superpixels |
| **Theoretical foundation** | Local approximation | CNN gradients | Game theory |

#### For tabular data:

| Criterion | LIME Tabular | SHAP TreeExplainer | Feature Importance |
|-----------|--------------|-------------------|-------------------|
| **Type** | Model-agnostic | Model-specific (trees) | Model-specific |
| **Required access** | predict() function | sklearn model | sklearn model |
| **Time** | ~2s | ~0.5s | ~0.01s |
| **Granularity** | Per feature | Per feature | Per feature (global) |
| **Foundation** | Local approximation | Exact Shapley values | Gini impurity reduction |

---

## 6. Data Processing Pipeline

### 6.1 Audio Pipeline

**File**: `utils/audio_processing.py`

```
.wav file → librosa.load() → Mel-spectrogram → PNG image → Resize 224x224 → Normalization [0,1]
```

**Detailed steps**:

```python
def create_spectrogram_from_upload(uploaded_file, temp_dir):
    # 1. Save uploaded file
    audio_path = os.path.join(temp_dir, uploaded_file.name)
    with open(audio_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # 2. Load audio with librosa
    y, sr = librosa.load(audio_path)

    # 3. Compute mel spectrogram
    ms = librosa.feature.melspectrogram(y=y, sr=sr)

    # 4. Convert to decibels (logarithmic scale)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # 5. Display and save
    librosa.display.specshow(log_ms, sr=sr)
    plt.savefig(spec_path)

    # 6. Load and resize for model
    image_data = load_img(spec_path, target_size=(224, 224))

    return image_data, spec_path
```

**Why mel spectrogram?**
- **Perceptual representation**: The mel scale is non-linear and better corresponds to human perception of frequencies
- **Compression**: Reduces dimensionality while preserving relevant information
- **CNN compatibility**: Transforms a 1D signal (audio) into a 2D image analyzable by convolutional networks

### 6.2 Image Pipeline

**File**: `utils/image_processing.py`

```
.jpg/.png file → PIL.Image.open() → Convert RGB → Resize 224x224 → Normalization [0,1] → DenseNet preprocessing
```

**Detailed steps**:

```python
def load_image_from_upload(uploaded_file, temp_dir, target_size=(224, 224)):
    # 1. Save file
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # 2. Load and convert to RGB (handles grayscale images)
    image = Image.open(image_path).convert('RGB')

    # 3. Resize
    image_resized = image.resize(target_size)

    return image_resized, image_path

def image_to_array(image_data, normalize=True):
    img_array = np.array(image_data)
    if normalize:
        img_array = img_array / 255.0
    return img_array
```

**DenseNet preprocessing**:
```python
# Applied in ImageClassifier.predict()
processed = tf.keras.applications.densenet.preprocess_input(image_array * 255.0)
```

This preprocessing centers pixels according to ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] approximately).

### 6.3 CSV Pipeline (tabular data)

**File**: `utils/tabular_processing.py`

```
.csv file → pandas.read_csv() → Column validation → Feature extraction → Numpy array
```

**Detailed steps**:

```python
def load_csv_from_upload(uploaded_file):
    # 1. Load CSV with pandas
    df = pd.read_csv(uploaded_file)
    return df

def validate_fraud_csv(df):
    # 2. Check required columns
    required_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    missing = set(required_columns) - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "OK"

def get_sample_for_analysis(df, row_index=0):
    # 3. Extract a row for analysis
    feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
    features = df[feature_columns].iloc[row_index].values
    return features
```

**Expected format**:
The CSV file must follow the Kaggle Credit Card Fraud dataset format:
- Columns V1 to V28: Anonymized features (PCA result)
- Amount column: Transaction amount
- Class column (optional): Real label for validation

---

## 7. User Interface

### 7.1 Navigation

**File**: `app.py`

```python
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Comparison"])

if page == "Home":
    render_home_page()
elif page == "Comparison":
    render_comparison_page()
```

### 7.2 Home Page

**File**: `views/home.py`

**Layout**:
```
+-------------------------------------------------------------+
|                    Unified XAI Platform                      |
+------------------------+------------------------------------+
|     Upload File        |       Configuration                 |
|  [File uploader]       |  Model: [Dropdown]                  |
|  [Preview audio/image] |  XAI: [Multiselect]                 |
+------------------------+------------------------------------+
|                  [Run Analysis Button]                       |
+-------------------------------------------------------------+
|  Results:                                                    |
|  - Spectrogram / Image preview                               |
|  - Classification: REAL/FAKE or BENIGN/MALIGNANT             |
|  - Confidence: XX%                                           |
+-------------------------------------------------------------+
|  XAI Explanations:                                           |
|  [LIME visualization]                                        |
|  [Grad-CAM visualization]                                    |
|  [SHAP visualization]                                        |
+-------------------------------------------------------------+
```

**Automatic type detection**:
```python
def detect_input_type(file):
    file_ext = file.name.lower().split('.')[-1]
    if file_ext in ['wav', 'mp3', 'flac', 'ogg']:
        return 'audio'
    elif file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
        return 'image'
    elif file_ext == 'csv':
        return 'tabular'
    return None
```

### 7.3 Comparison Page

**File**: `views/comparison.py`

**Layout**:
```
+-------------------------------------------------------------+
|                    XAI Comparison                            |
+-------------------------------------------------------------+
|  Current Analysis: Audio | MobileNet | Prediction: REAL      |
+-------------------------------------------------------------+
|  Original Input: [Spectrogram/Image]                         |
|  Classification: REAL (85.2%)                                |
+-------------------------------------------------------------+
|  Select XAI Methods: [LIME] [Grad-CAM] [SHAP]                |
|                  [Run Comparison Button]                     |
+--------------+--------------+-------------------------------+
|    LIME      |   Grad-CAM   |    SHAP                       |
|  [Visual]    |   [Visual]   |   [Visual]                    |
|  Time: 10.2s |  Time: 1.1s  |  Time: 32.5s                  |
+--------------+--------------+-------------------------------+
|  Summary Table:                                              |
|  | Method | Time | Speed | Type |                           |
|  | LIME   | 10s  | Medium| Model-agnostic |                 |
|  | Grad-CAM| 1s  | Fast  | Gradient-based |                 |
|  | SHAP   | 32s  | Slow  | Model-agnostic |                 |
+-------------------------------------------------------------+
```

---

## 8. Challenges Encountered and Solutions

### 8.1 Missing Pre-trained Model for Images

**Problem**: The LungCancerDetection repository contains only documentation, no code or models.

**Solution**: We implemented DenseNet121 with ImageNet weights. This is a demonstration of XAI architecture, not a clinical tool. For real use, fine-tuning on the CheXpert dataset would be necessary.

**Justification**: The professor clarified that the objective is to merge repositories, not necessarily to train models. Reusing pre-trained models with citation is acceptable.

### 8.2 Python 3.11 Compatibility

**Problem**: TensorFlow 2.6.2 (original repository version) is not compatible with Python 3.11.

**Solution**: Update to TensorFlow 2.15.0 and adapt other dependencies:
```
tensorflow>=2.15.0
keras>=3.0.0
streamlit>=1.28.0
```

### 8.3 Session State Lost Between Pages

**Problem**: Analysis results were not available on the comparison page.

**Solution**: Explicit save in `st.session_state` after each analysis:
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

### 8.4 Grad-CAM for Audio

**Problem**: MobileNet does not have a convolutional layer easily exploitable for Grad-CAM.

**Solution**: Following the original repository approach, we use VGG16 for audio Grad-CAM. Although not the same model as for classification, this allows visualizing important spectrogram regions.

### 8.5 SHAP Computation Time

**Problem**: SHAP can take more than a minute per image.

**Solution**: Reduced parameters for the comparison page:
- `n_segments=30` instead of 50
- `nsamples=50` instead of 100

The home page uses full parameters for more precision.

---

## 9. Improvements Over Original Repositories

### 9.1 Unified Interface

| Before | After |
|--------|-------|
| Two separate repositories | One single application |
| Audio only (Streamlit) | Audio + Image |
| No interface for images | Complete interface |

### 9.2 Extended XAI Coverage

| Method | Audio Repo | Image Repo | Our Platform |
|--------|------------|------------|--------------|
| LIME | Yes | No | Yes (audio + image + tabular) |
| Grad-CAM | Yes | Yes (doc) | Yes (audio + image) |
| SHAP | Notebooks | No | Yes (audio + image + tabular) |
| Feature Importance | No | No | Yes (tabular) |

### 9.3 Comparison Page

**Feature absent from original repos**:
- Side-by-side visualization of 2-3 XAI methods
- Computation time metrics
- Comparative characteristics table

### 9.4 Tabular Data Support

**Bonus feature implemented**:
- Fraud detection via CSV file
- 3 dedicated XAI methods: LIME Tabular, SHAP TreeExplainer, Feature Importance
- Test file provided (`file_test/fraud_test_data.csv`)

### 9.5 Modernized Code

- Python 3.11+ compatible
- Up-to-date dependencies (TensorFlow 2.15, Streamlit 1.28)
- Modular and maintainable architecture

---

## 10. Limitations and Perspectives

### 10.1 Current Limitations

**Image model not fine-tuned**:
Our ImageClassifier uses ImageNet weights. Predictions on X-rays are not reliable. This is a demonstration of architecture, not a medical tool.

**Indirect audio Grad-CAM**:
Using VGG16 instead of MobileNet for audio Grad-CAM is a compromise. Visualizations show important regions according to VGG16, not the actual classification model.

### 10.2 Possible Improvements

1. **Fine-tune image model** on CheXpert for realistic medical predictions
2. **Add other audio models** (VGG16, ResNet, Custom CNN available in original repo)
3. **PDF export** of analysis results
4. **Model caching** to speed up repeated analyses
5. **Unit tests** for preprocessing modules and classifiers
6. **Train fraud model** on real Kaggle dataset for better predictions

### 10.3 Research Directions

- **Quantitative comparison** of XAI methods (fidelity, stability, sensitivity)
- **Additional XAI methods**: Integrated Gradients, Layer-wise Relevance Propagation
- **Multi-modal analysis**: combine audio and image for the same prediction
- **Cloud deployment**: AWS/GCP for online demonstration

---

## Appendices

### A. Main Dependencies

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

### B. Useful Commands

```bash
# Launch application
streamlit run app.py

# Launch with specific port
streamlit run app.py --server.port 8080

# Debug mode
streamlit run app.py --logger.level debug
```

### C. session_state Structure

```python
st.session_state = {
    'uploaded_file': <UploadedFile>,
    'input_type': 'audio' | 'image' | 'tabular',
    'selected_model': 'mobilenet' | 'densenet121' | 'randomforest_fraud',
    'selected_xai': ['lime', 'gradcam', 'shap', 'lime_tabular', 'shap_tabular', 'feature_importance'],
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
