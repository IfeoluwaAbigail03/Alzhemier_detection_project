import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io
import matplotlib.pyplot as plt
import json
import base64

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('alzheimer_resnet50.keras')
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# ── Load demo results ─────────────────────────────────────────────────────
@st.cache_data
def load_demo():
    try:
        with open('alzheimer_demo_results.json', 'r') as f:
            return json.load(f)
    except:
        return []

model        = load_model()
demo_results = load_demo()

# ── Constants ─────────────────────────────────────────────────────────────
IMG_SIZE = 192
CLASSES  = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

CLASS_INFO = {
    'NonDemented'     : {'color': '#1D9E75', 'icon': '✅', 'severity': 0,
                         'description': 'No signs of Alzheimer\'s detected. Brain structure appears normal.'},
    'VeryMildDemented': {'color': '#EF9F27', 'icon': '⚠️', 'severity': 1,
                         'description': 'Very mild cognitive decline. Early intervention recommended.'},
    'MildDemented'    : {'color': '#D85A30', 'icon': '🔶', 'severity': 2,
                         'description': 'Mild dementia detected. Medical consultation strongly advised.'},
    'ModerateDemented': {'color': '#A32D2D', 'icon': '🔴', 'severity': 3,
                         'description': 'Moderate dementia detected. Immediate medical attention required.'}
}

# ── Prediction function ───────────────────────────────────────────────────
def predict(img_array):
    img_resized   = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_processed = preprocess_input(
        np.expand_dims(img_resized.astype(np.float32), axis=0)
    )
    preds      = model.predict(img_processed, verbose=0)[0]
    pred_class = CLASSES[np.argmax(preds)]
    confidence = float(np.max(preds))
    probs      = {c: float(p) for c, p in zip(CLASSES, preds)}
    return pred_class, confidence, probs

# ── Grad-CAM function ─────────────────────────────────────────────────────
def get_gradcam(img_array):
    try:
        img_resized   = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_processed = preprocess_input(
            np.expand_dims(img_resized.astype(np.float32), axis=0)
        )

        last_conv = None
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                for inner in reversed(layer.layers):
                    if isinstance(inner, tf.keras.layers.Conv2D):
                        last_conv = inner
                        break
            if last_conv:
                break

        if last_conv is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer
                    break

        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_processed)
            loss = preds[:, np.argmax(preds[0])]

        grads       = tape.gradient(loss, conv_out)
        pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap     = tf.reduce_mean(conv_out[0] * pooled, axis=-1).numpy()
        heatmap     = np.maximum(heatmap, 0)
        heatmap     = heatmap / (heatmap.max() + 1e-8)
        heatmap     = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_col = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_col = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)
        overlay     = cv2.addWeighted(img_resized, 0.6, heatmap_col, 0.4, 0)
        return overlay

    except Exception as e:
        st.warning(f"Grad-CAM error: {e}")
        return None

# ── Header ────────────────────────────────────────────────────────────────
st.title("🧠 Alzheimer's MRI Classifier")
st.markdown("Upload a brain MRI scan to classify Alzheimer's severity using **ResNet50 transfer learning** trained on 44,000+ MRI images.")
st.divider()

# ── Model metrics ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Accuracy",  "97.8%",    "on 6,600 test images")
col2.metric("Model",          "ResNet50",  "ImageNet transfer learning")
col3.metric("Training data",  "44,000+",   "MRI scans")
col4.metric("Classes",        "4",         "severity levels")

st.divider()

# ── Severity scale ────────────────────────────────────────────────────────
st.subheader("📊 Alzheimer's Severity Scale")
cols = st.columns(4)
for i, (cls, info) in enumerate(CLASS_INFO.items()):
    with cols[i]:
        st.markdown(f"""
        <div style='background:{info["color"]}22;border-left:4px solid {info["color"]};
             padding:10px;border-radius:4px;margin-bottom:10px;'>
        <b>{info["icon"]} {cls}</b><br>
        <small>{info["description"]}</small>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Sample MRI images per class ───────────────────────────────────────────
st.subheader("🖼️ Sample MRI Scans — What Each Stage Looks Like")
st.markdown("Real MRI scans from the test set, classified by the model with confidence scores:")

class_order = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

if demo_results:
    img_cols = st.columns(4)
    for i, class_name in enumerate(class_order):
        sample = next((r for r in demo_results if r['true_class'] == class_name), None)
        if sample:
            with img_cols[i]:
                info      = CLASS_INFO[class_name]
                img_bytes = base64.b64decode(sample['image_b64'])
                img       = Image.open(io.BytesIO(img_bytes))
                st.image(img, use_container_width=True)
                st.markdown(f"""
                <div style='background:{info["color"]}22;
                     border-left:3px solid {info["color"]};
                     padding:8px;border-radius:4px;text-align:center;'>
                <b>{info["icon"]} {class_name}</b><br>
                <small>Confidence: {sample["confidence"]*100:.1f}%</small><br>
                <small style='color:#666;'>{info["description"]}</small>
                </div>
                """, unsafe_allow_html=True)

    # Show all 12 samples in expandable section
    with st.expander("📋 View all 12 sample predictions"):
        for row_start in range(0, len(demo_results), 4):
            batch = demo_results[row_start:row_start+4]
            batch_cols = st.columns(4)
            for j, result in enumerate(batch):
                with batch_cols[j]:
                    info      = CLASS_INFO[result['pred_class']]
                    img_bytes = base64.b64decode(result['image_b64'])
                    img       = Image.open(io.BytesIO(img_bytes))
                    correct   = result['correct']
                    status    = "✅" if correct else "❌"
                    st.image(img, use_container_width=True)
                    st.markdown(f"""
                    <div style='background:{info["color"]}22;
                         border-left:3px solid {info["color"]};
                         padding:6px;border-radius:4px;font-size:11px;'>
                    {status} <b>{result["pred_class"]}</b><br>
                    True: {result["true_class"]}<br>
                    Conf: {result["confidence"]*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)

st.divider()

# ── Upload section ────────────────────────────────────────────────────────
st.subheader("🔬 Upload Your Own MRI Scan for Classification")

if model is None:
    st.error("Model failed to load. Please check deployment logs.")
else:
    uploaded_file = st.file_uploader(
        "Upload a brain MRI image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col_img, col_result = st.columns(2)

        with col_img:
            st.image(img_rgb, caption="Uploaded MRI Scan",
                     use_container_width=True)

        with col_result:
            with st.spinner("Analysing MRI scan..."):
                pred_class, confidence, probs = predict(img_rgb)

            info = CLASS_INFO[pred_class]

            st.markdown(f"""
            <div style='background:{info["color"]}22;border-left:6px solid {info["color"]};
                 padding:16px;border-radius:6px;margin-bottom:16px;'>
            <h3 style='margin:0;color:{info["color"]};'>{info["icon"]} {pred_class}</h3>
            <p style='margin:8px 0 4px;'><b>Confidence:</b> {confidence*100:.1f}%</p>
            <p style='margin:0;font-size:14px;'>{info["description"]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Class Probabilities")
            fig, ax = plt.subplots(figsize=(6, 3))
            colors  = [CLASS_INFO[c]['color'] for c in CLASSES]
            values  = [probs[c] for c in CLASSES]
            bars    = ax.barh(CLASSES, values, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val*100:.1f}%', va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("🔥 Grad-CAM — Brain Regions Driving the Prediction")
        with st.spinner("Generating Grad-CAM heatmap..."):
            gradcam = get_gradcam(img_rgb)

        if gradcam is not None:
            col_orig, col_cam = st.columns(2)
            with col_orig:
                st.image(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)),
                         caption="Original MRI",
                         use_container_width=True)
            with col_cam:
                st.image(gradcam,
                         caption="Grad-CAM — highlighted regions influence prediction",
                         use_container_width=True)
        else:
            st.info("Grad-CAM visualisation not available for this image.")

    else:
        st.info("👆 Upload an MRI scan above to get a prediction.")

st.divider()

# ── Performance table ─────────────────────────────────────────────────────
st.subheader("📈 Model Performance")
st.markdown("""
| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| MildDemented | 96.9% | 99.6% | 98.2% |
| ModerateDemented | 99.9% | 99.9% | 99.9% |
| NonDemented | 99.2% | 94.4% | 96.7% |
| VeryMildDemented | 95.5% | 98.4% | 96.9% |
| **Overall** | **97.9%** | **98.1%** | **97.9%** |
""")

st.divider()

# ── Architecture ──────────────────────────────────────────────────────────
st.subheader("🏗️ Model Architecture")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    **Transfer Learning Pipeline:**
    - Base: ResNet50 pretrained on ImageNet
    - GlobalAveragePooling2D
    - Dense(512, ReLU) + BatchNorm + Dropout(0.5)
    - Dense(256, ReLU) + BatchNorm + Dropout(0.3)
    - Dense(4, Softmax)
    """)

with col_b:
    st.markdown("""
    **Training Setup:**
    - Image size: 192×192
    - Batch size: 32
    - Learning rate: 0.0001
    - Epochs: 15 with early stopping
    - Data augmentation: rotation, flip, zoom, shift
    - Callbacks: EarlyStopping, ReduceLROnPlateau
    """)

st.divider()
st.warning("""
⚕️ **Clinical Disclaimer:** This tool is for research and educational
purposes only. It is not a substitute for professional medical diagnosis.
Always consult a qualified neurologist for medical decisions.
""")
st.caption("Built with Python · TensorFlow 2.19 · ResNet50 · Keras 3.10 · Streamlit | Ifeoluwa Abigail Oyedemi")