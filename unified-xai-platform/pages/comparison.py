"""
Comparison Page
Side-by-side comparison of different XAI techniques.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic
import cv2
import tensorflow as tf
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_processing import spectrogram_to_array
from utils.image_processing import image_to_array


# =============================================================================
# IMAGE/AUDIO XAI COMPARISON FUNCTIONS
# =============================================================================

def run_lime_comparison(image_data, model, input_type, is_classifier=False):
    """Run LIME and return figure + computation time."""
    from lime import lime_image

    start_time = time.time()

    if input_type == 'audio':
        img_array = spectrogram_to_array(image_data, normalize=True)
        predict_fn = model.predict
    else:
        img_array = image_to_array(image_data, normalize=True)
        if is_classifier:
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype('float64'),
        predict_fn,
        hide_color=0,
        num_samples=500
    )

    prediction = predict_fn(np.expand_dims(img_array, axis=0))
    class_label = np.argmax(prediction[0])

    temp, mask = explanation.get_image_and_mask(
        class_label,
        positive_only=False,
        num_features=8,
        hide_rest=True
    )

    computation_time = time.time() - start_time

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title("LIME Explanation")
    ax.axis('off')
    plt.tight_layout()

    return fig, computation_time


def run_gradcam_comparison(image_data, model, class_idx, input_type, is_classifier=False):
    """Run Grad-CAM and return figure + computation time."""
    from keras.preprocessing.image import img_to_array

    start_time = time.time()

    if input_type == 'audio':
        img_array = img_to_array(image_data)
        x = np.expand_dims(img_array, axis=0)
        x = tf.keras.applications.vgg16.preprocess_input(x)

        vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
        last_conv_layer = vgg_model.get_layer('block5_conv3')
        grad_model = tf.keras.models.Model([vgg_model.inputs], [last_conv_layer.output, vgg_model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(x)
            class_output = preds[:, class_idx]

        grads = tape.gradient(class_output, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmap = cv2.resize(np.float32(heatmap), (224, 224))
        heatmap_colored = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

        original_img = np.array(image_data)
        if original_img.shape[:2] != (224, 224):
            original_img = cv2.resize(original_img, (224, 224))
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    else:
        img_array = image_to_array(image_data, normalize=True)
        x = np.expand_dims(img_array, axis=0)

        keras_model = model.get_model()
        last_conv_layer_name = model.get_last_conv_layer_name()

        x_processed = tf.keras.applications.densenet.preprocess_input(x * 255.0)

        grad_model = tf.keras.models.Model(
            inputs=keras_model.input,
            outputs=[keras_model.get_layer(last_conv_layer_name).output, keras_model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(x_processed)
            class_output = preds[:, class_idx]

        grads = tape.gradient(class_output, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

        heatmap_resized = cv2.resize(np.float32(heatmap), (224, 224))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        original_img = np.array(image_data)
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    computation_time = time.time() - start_time

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(superimposed)
    ax.set_title("Grad-CAM Explanation")
    ax.axis('off')
    plt.tight_layout()

    return fig, computation_time


def run_shap_comparison(image_data, model, input_type, is_classifier=False):
    """Run SHAP and return figure + computation time."""
    import shap

    start_time = time.time()

    if input_type == 'audio':
        img_array = spectrogram_to_array(image_data, normalize=True)
        predict_fn_base = model.predict
    else:
        img_array = image_to_array(image_data, normalize=True)
        if is_classifier:
            predict_fn_base = model.predict_proba
        else:
            predict_fn_base = model.predict

    prediction = predict_fn_base(np.expand_dims(img_array, axis=0))
    class_label = np.argmax(prediction[0])

    img_uint8 = (img_array * 255).astype(np.uint8)
    segments = slic(img_uint8, n_segments=30, compactness=10, sigma=1)

    unique_segments = np.unique(segments)
    n_segments = len(unique_segments)

    def mask_image(mask, img, segs, seg_ids, background=0.0):
        masked = img.copy()
        for idx, keep in enumerate(mask):
            if not keep:
                seg_id = seg_ids[idx]
                masked[segs == seg_id] = background
        return masked

    def predict_fn(masks):
        preds = []
        for mask in masks:
            masked_img = mask_image(mask, img_array, segments, unique_segments)
            masked_img = np.expand_dims(masked_img, axis=0)
            pred = predict_fn_base(masked_img)
            preds.append(pred[0])
        return np.array(preds)

    background = np.ones((1, n_segments))
    explainer = shap.KernelExplainer(predict_fn, background)

    test_mask = np.ones((1, n_segments))
    shap_values = explainer.shap_values(test_mask, nsamples=50)

    if isinstance(shap_values, list):
        values = shap_values[class_label][0]
    else:
        values = shap_values[0]

    heatmap = np.zeros(segments.shape, dtype=np.float64)
    for idx, val in enumerate(values):
        if idx < len(unique_segments):
            seg_id = unique_segments[idx]
            if hasattr(val, '__len__') and len(val) > 1:
                scalar_val = float(val[class_label])
            else:
                scalar_val = float(val)
            heatmap[segments == seg_id] = scalar_val

    if heatmap.max() != heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    computation_time = time.time() - start_time

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_data)
    ax.imshow(heatmap, cmap='RdBu_r', alpha=0.5)
    ax.set_title("SHAP Explanation")
    ax.axis('off')
    plt.tight_layout()

    return fig, computation_time


# =============================================================================
# TABULAR XAI COMPARISON FUNCTIONS
# =============================================================================

def run_lime_tabular_comparison(X, classifier, feature_names, class_names):
    """Run LIME Tabular and return figure + computation time."""
    from lime.lime_tabular import LimeTabularExplainer

    start_time = time.time()

    prediction = classifier.predict_proba(X)
    class_label = np.argmax(prediction[0])

    # Generate synthetic training data for LIME
    np.random.seed(42)
    n_samples = 500
    n_features = X.shape[1]
    training_data = np.random.randn(n_samples, n_features)
    training_data = np.vstack([X, training_data])

    explainer = LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        X[0],
        classifier.predict_proba,
        num_features=min(10, len(feature_names)),
        num_samples=1000
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    feature_weights = exp.as_list()
    features = [fw[0][:15] + '...' if len(fw[0]) > 15 else fw[0] for fw in feature_weights]
    weights = [fw[1] for fw in feature_weights]

    colors = ['green' if w > 0 else 'red' for w in weights]
    y_pos = np.arange(len(features))

    ax.barh(y_pos, weights, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('Contribution')
    ax.set_title(f'LIME Tabular')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    computation_time = time.time() - start_time

    return fig, computation_time


def run_shap_tabular_comparison(X, classifier, feature_names, class_names):
    """Run SHAP TreeExplainer and return figure + computation time."""
    import shap

    start_time = time.time()

    model = classifier.get_model()
    prediction = classifier.predict_proba(X)
    class_label = np.argmax(prediction[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        values = shap_values[class_label][0]
    else:
        if shap_values.ndim == 3:
            values = shap_values[0, :, class_label]
        else:
            values = shap_values[0]

    fig, ax = plt.subplots(figsize=(5, 5))

    indices = np.argsort(np.abs(values))[::-1][:10]
    sorted_features = [feature_names[i][:12] + '..' if len(feature_names[i]) > 12 else feature_names[i] for i in indices]
    sorted_values = values[indices]

    colors = ['#ff0051' if v < 0 else '#008bfb' for v in sorted_values]
    y_pos = np.arange(len(sorted_features))

    ax.barh(y_pos, sorted_values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=8)
    ax.set_xlabel('SHAP Value')
    ax.set_title('SHAP TreeExplainer')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    computation_time = time.time() - start_time

    return fig, computation_time


def run_feature_importance_comparison(classifier, feature_names):
    """Run Feature Importance and return figure + computation time."""
    start_time = time.time()

    importance_dict = classifier.get_feature_importance()

    fig, ax = plt.subplots(figsize=(5, 5))

    if not importance_dict:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, 0.0

    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    features = [item[0][:12] + '..' if len(item[0]) > 12 else item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]

    y_pos = np.arange(len(features))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))

    ax.barh(y_pos, importances, color=colors[::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()

    plt.tight_layout()
    computation_time = time.time() - start_time

    return fig, computation_time


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_comparison_page():
    """Render the comparison page."""

    st.title("XAI Comparison")
    st.markdown("Compare results from different explainability methods side by side")

    st.markdown("---")

    # Check if there are any results to compare
    if 'analysis_results' not in st.session_state:
        st.warning("No analysis results available yet.")
        st.info("Please go to the **Home** page, upload a file, and run an analysis first.")
        return

    results = st.session_state['analysis_results']
    input_type = results.get('input_type', 'unknown')
    model_name = results.get('model', 'unknown')
    class_label = results.get('class_label', 0)

    # Check required fields based on input type
    if input_type in ['audio', 'image'] and 'image_data' not in results:
        st.warning("No image data available. Please run a new analysis on the Home page.")
        return
    elif input_type == 'tabular' and 'tabular_data' not in results:
        st.warning("No tabular data available. Please run a new analysis on the Home page.")
        return

    # Display current analysis info
    st.subheader("Current Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Input Type", input_type.capitalize())

    with col2:
        st.metric("Model", model_name.upper())

    with col3:
        if input_type == 'audio':
            class_names = ['Real', 'Fake']
        elif input_type == 'image':
            class_names = ['Benign', 'Malignant']
        else:
            class_names = ['Legitimate', 'Fraud']
        st.metric("Prediction", class_names[class_label])

    st.markdown("---")

    # Show original input based on type
    st.subheader("Original Input")

    if input_type in ['audio', 'image']:
        image_data = results['image_data']
        col_orig, col_info = st.columns([1, 1])
        with col_orig:
            if input_type == 'audio':
                st.image(image_data, caption="Mel-Spectrogram", width=300)
            else:
                st.image(image_data, caption="X-Ray Image", width=300)

        with col_info:
            st.markdown("**Classification Result:**")
            confidence = results.get('prediction', [[0.5, 0.5]])[0][class_label] * 100
            st.write(f"- **Class:** {class_names[class_label]}")
            st.write(f"- **Confidence:** {confidence:.2f}%")

    else:  # tabular
        feature_names = results.get('feature_names', [])
        sample_df = results.get('sample_df')
        if sample_df is not None:
            st.dataframe(sample_df[feature_names].round(4) if feature_names else sample_df, width="stretch")

        col1, col2 = st.columns(2)
        prediction = results.get('prediction', [[0.5, 0.5]])
        with col1:
            st.metric("Legitimate Probability", f"{prediction[0][0]*100:.2f}%")
        with col2:
            st.metric("Fraud Probability", f"{prediction[0][1]*100:.2f}%")

    st.markdown("---")

    # XAI Method Selection for Comparison - based on MODEL
    st.subheader("Select XAI Methods to Compare")

    # Define method names and compatibility
    method_names = {
        'lime': 'LIME',
        'gradcam': 'Grad-CAM',
        'shap': 'SHAP',
        'lime_tabular': 'LIME Tabular',
        'shap_tabular': 'SHAP TreeExplainer',
        'feature_importance': 'Feature Importance'
    }

    MODEL_XAI_COMPATIBILITY = {
        'mobilenet': ['lime', 'gradcam', 'shap'],
        'densenet121': ['lime', 'gradcam', 'shap'],
        'randomforest_fraud': ['lime_tabular', 'shap_tabular', 'feature_importance']
    }

    available_methods = MODEL_XAI_COMPATIBILITY.get(model_name, [])

    # Select all methods by default (same as home page)
    selected_methods = st.multiselect(
        "Choose 2 or 3 methods to compare:",
        options=available_methods,
        format_func=lambda x: method_names.get(x, x),
        default=available_methods  # All methods selected by default
    )

    if len(selected_methods) < 2:
        st.info("Please select at least 2 XAI methods to enable comparison.")
        return

    # Run Comparison Button
    if st.button("Run Comparison", type="primary", width="stretch"):

        st.markdown("---")
        st.subheader("Side-by-Side Comparison")

        # Load the appropriate model
        try:
            if input_type == 'audio':
                from models.audio.audio_classifier import AudioClassifier
                classifier = AudioClassifier()
                model = classifier.get_model()
                is_classifier = False
            elif input_type == 'image':
                from models.image.image_classifier import ImageClassifier
                classifier = ImageClassifier()
                model = classifier
                is_classifier = True
            else:  # tabular
                from models.tabular.fraud_classifier import FraudClassifier
                classifier = FraudClassifier()
                model = classifier
                is_classifier = True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

        # Create columns for comparison
        cols = st.columns(len(selected_methods))
        computation_times = {}

        for idx, method in enumerate(selected_methods):
            with cols[idx]:
                st.markdown(f"### {method_names.get(method, method)}")

                with st.spinner(f"Computing {method_names.get(method, method)}..."):
                    try:
                        if input_type in ['audio', 'image']:
                            image_data = results['image_data']
                            if method == 'lime':
                                fig, comp_time = run_lime_comparison(
                                    image_data, model if not is_classifier else classifier,
                                    input_type, is_classifier
                                )
                            elif method == 'gradcam':
                                fig, comp_time = run_gradcam_comparison(
                                    image_data, model if not is_classifier else classifier,
                                    class_label, input_type, is_classifier
                                )
                            elif method == 'shap':
                                fig, comp_time = run_shap_comparison(
                                    image_data, model if not is_classifier else classifier,
                                    input_type, is_classifier
                                )
                        else:  # tabular
                            X = results['tabular_data']
                            feature_names = results.get('feature_names', classifier.get_feature_names())

                            if method == 'lime_tabular':
                                fig, comp_time = run_lime_tabular_comparison(
                                    X, classifier, feature_names, class_names
                                )
                            elif method == 'shap_tabular':
                                fig, comp_time = run_shap_tabular_comparison(
                                    X, classifier, feature_names, class_names
                                )
                            elif method == 'feature_importance':
                                fig, comp_time = run_feature_importance_comparison(
                                    classifier, feature_names
                                )

                        st.pyplot(fig)
                        plt.close(fig)
                        computation_times[method] = comp_time

                        st.caption(f"Computation time: {comp_time:.2f}s")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        computation_times[method] = None

        # Comparison Summary
        st.markdown("---")
        st.subheader("Comparison Summary")

        # Create summary table
        summary_data = []
        for method in selected_methods:
            comp_time = computation_times.get(method)
            if comp_time is not None:
                speed = "Fast" if comp_time < 5 else ("Medium" if comp_time < 15 else "Slow")

                if method in ['lime', 'shap', 'lime_tabular', 'shap_tabular']:
                    method_type = "Model-agnostic"
                elif method == 'gradcam':
                    method_type = "Gradient-based"
                else:
                    method_type = "Model-specific"

                summary_data.append({
                    "Method": method_names.get(method, method),
                    "Computation Time": f"{comp_time:.2f}s",
                    "Speed": speed,
                    "Type": method_type
                })

        if summary_data:
            df = pd.DataFrame(summary_data)
            st.table(df)

        # Method descriptions
        st.markdown("---")
        st.subheader("Method Descriptions")

        method_descriptions = {
            'lime': """
            **LIME (Local Interpretable Model-agnostic Explanations)**
            - Creates local approximations around the prediction
            - Highlights superpixels that contribute to the decision
            - Model-agnostic: works with any classifier
            """,
            'gradcam': """
            **Grad-CAM (Gradient-weighted Class Activation Mapping)**
            - Uses gradients from the last convolutional layer
            - Creates a heatmap showing important regions
            - Specific to CNN architectures
            """,
            'shap': """
            **SHAP (SHapley Additive exPlanations)**
            - Based on game theory (Shapley values)
            - Assigns importance scores to each feature/segment
            - Provides theoretically grounded explanations
            """,
            'lime_tabular': """
            **LIME Tabular**
            - Explains individual predictions for tabular data
            - Creates local linear approximations
            - Shows which features pushed the prediction in each direction
            """,
            'shap_tabular': """
            **SHAP TreeExplainer**
            - Optimized SHAP implementation for tree-based models
            - Exact computation of Shapley values
            - Shows contribution of each feature to the prediction
            """,
            'feature_importance': """
            **Feature Importance**
            - Global importance scores from the model
            - Shows which features are most influential overall
            - Based on decrease in impurity (Gini importance)
            """
        }

        for method in selected_methods:
            if method in method_descriptions:
                with st.expander(f"{method_names.get(method, method)} Details"):
                    st.markdown(method_descriptions[method])

        st.success("Comparison complete!")
