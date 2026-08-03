# Alzheimer's MRI Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![Accuracy](https://img.shields.io/badge/Accuracy-97.8%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

## Live App
**[View the live app on Streamlit](https://alzhemierdetectionproject-7m4vf8e7xhdhpubcrycorx.streamlit.app/)**

---

## Project Summary
An interactive brain MRI classification tool that detects Alzheimer's
disease severity using **ResNet50 transfer learning** trained on 44,000+
MRI images. Upload any brain MRI scan to get an instant prediction,
confidence score, and Grad-CAM heatmap showing which brain regions
drove the classification.

> *"Which stage of Alzheimer's does this brain MRI indicate — and
> which regions of the brain show the most significant changes?"*

---

##  Key Results

| Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| MildDemented | 96.9% | 99.6% | 98.2% | 1,500 |
| ModerateDemented | 99.9% | 99.9% | 99.9% | 1,500 |
| NonDemented | 99.2% | 94.4% | 96.7% | 1,920 |
| VeryMildDemented | 95.5% | 98.4% | 96.9% | 1,680 |
| **Overall** | **97.9%** | **98.1%** | **97.9%** | **6,600** |

**Test Accuracy: 97.8%** on 6,600 held-out MRI images.

---

## Alzheimer's Severity Scale

| Stage | Description |
|---|---|
| NonDemented | No signs detected. Brain structure appears normal. |
| VeryMildDemented | Very mild cognitive decline. Early intervention recommended. |
| MildDemented | Mild dementia detected. Medical consultation strongly advised. |
| ModerateDemented | Moderate dementia. Immediate medical attention required. |

---

## Features
- **Live MRI upload** — upload any brain MRI and get instant classification
- **Grad-CAM heatmaps** — visualise which brain regions influence the prediction
- **Confidence scores** — probability breakdown across all 4 severity classes
- **Sample gallery** — pre-classified MRI scans showing what each stage looks like
- **Clinical context** — plain-English interpretation of each prediction

---

## Model Architecture
Input (192×192×3)
↓
ResNet50 (pretrained on ImageNet — frozen base)
↓
GlobalAveragePooling2D
↓
Dense(512, ReLU) → BatchNorm → Dropout(0.5)
↓
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
↓
Dense(4, Softmax) → class probabilities

**Training details:**
- Dataset: 44,000+ augmented MRI images
- Train/Val/Test split: 70% / 15% / 15%
- Learning rate: 0.0001 (Adam)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Data augmentation: rotation, zoom, flip, shift

---

## Dataset
- **Source:** [Alzheimer's Multiclass Dataset](https://www.kaggle.com/datasets/alzheimers-multiclass-dataset-equal-and-augmented) — Kaggle
- **Size:** 44,000+ MRI images across 4 classes
- **Format:** RGB images resized to 192×192

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| TensorFlow 2.19 / Keras 3.10 | Model training + inference |
| ResNet50 | Pretrained base model (ImageNet) |
| OpenCV | Image preprocessing |
| Grad-CAM | Model explainability |
| Streamlit | Interactive web app |
| Streamlit Cloud | Deployment |

---

## Run Locally

```bash
git clone https://github.com/IfeoluwaAbigail03/Alzhemier_detection_project.git
cd Alzhemier_detection_project
pip install -r requirements.txt
streamlit run app.py
```

---

##  Why This Project Matters
Alzheimer's disease affects 55 million people globally. Early detection
is critical — patients diagnosed at the VeryMild stage have significantly
better outcomes than those diagnosed at Moderate stage. This tool
demonstrates how deep learning can assist radiologists in faster,
more consistent MRI analysis.

The **Grad-CAM visualisation** is particularly important for clinical
trust — it shows exactly which brain regions (typically the hippocampus
and entorhinal cortex) the model focuses on, making the prediction
interpretable rather than a black box.

---

## Author
**Ifeoluwa Abigail Oyedemi**
-  [LinkedIn](https://www.linkedin.com/in/ifeoluwa-oyedemi/)
-  [GitHub](https://github.com/IfeoluwaAbigail03)
-  oyedemiifeoluwa03@gmail.com

---

 **Clinical Disclaimer:** This tool is for research and educational
purposes only. Not a substitute for professional medical diagnosis.
Always consult a qualified neurologist.