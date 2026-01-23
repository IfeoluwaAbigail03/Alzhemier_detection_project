Alzheimer's MRI Classification
A deep learning model that classifies Alzheimer's disease severity from MRI scans with 97.8% accuracy.

Quick Results
Accuracy: 97.8% on test data
Classes: 4 (Healthy → Advanced Alzheimer's)
Model: ResNet50 with transfer learning
Dataset: 44,000 MRI scans from Kaggle

Key Discovery
Found that brain scans get darker as Alzheimer's worsens:
Healthy: 77.5 brightness
Advanced: 69.5 brightness
10.3% intensity decrease from healthy to severe cases

Model Performance
Stage	Precision	Recall
Advanced	99.9%	99.9%
Noticeable	96.9%	99.6%
Very Early	95.5%	98.4%
Healthy	99.2%	94.4%
