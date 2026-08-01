import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('alzheimer_resnet50.keras')

model = load_model()

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
        # Find last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break
        if last_conv is None:
            # Try inner ResNet50 layers
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    for inner in reversed(layer.layers):
                        if isinstance(inner, tf.keras.layers.Conv2D):
                            last_conv = inner
                            break
                if last_conv:
                    break

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
        return None

# ── Header ────────────────────────────────────────────────────────────────
st.title("🧠 Alzheimer's MRI Classifier")
st.markdown("Upload a brain MRI scan to classify Alzheimer's severity using **ResNet50 transfer learning** trained on 44,000+ MRI images.")
st.divider()

# ── Model metrics ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Accuracy",  "97.8%",   "on 6,600 test images")
col2.metric("Model",          "ResNet50", "ImageNet transfer learning")
col3.metric("Training data",  "44,000+",  "MRI scans")
col4.metric("Classes",        "4",        "severity levels")

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

# ── Upload section ────────────────────────────────────────────────────────
st.subheader("🔬 Upload MRI Scan for Classification")
uploaded_file = st.file_uploader(
    "Upload a brain MRI image (JPG, PNG)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Read image
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

        # Probability chart
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

    # Grad-CAM
    st.subheader("🔥 Grad-CAM — Brain Regions Driving the Prediction")
    with st.spinner("Generating Grad-CAM heatmap..."):
        gradcam = get_gradcam(img_rgb)

    if gradcam is not None:
        col_orig, col_cam = st.columns(2)
        with col_orig:
            st.image(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)),
                     caption="Original MRI", use_container_width=True)
        with col_cam:
            st.image(gradcam,
                     caption="Grad-CAM Heatmap — highlighted regions influence prediction",
                     use_container_width=True)
    else:
        st.info("Grad-CAM not available for this image.")

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
st.warning("⚕️ **Clinical Disclaimer:** This tool is for research and educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified neurologist.")
st.caption("Built with Python · TensorFlow · ResNet50 · Streamlit | Ifeoluwa Abigail Oyedemi")