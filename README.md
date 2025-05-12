
# 🧠 DLOR Project – Traffic Sign Recognition using Deep Learning

![Python](https://img.shields.io/badge/Made%20with-Python-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg?logo=jupyter)
![Dataset](https://img.shields.io/badge/Dataset-GTSRB-yellowgreen)
![Stage](https://img.shields.io/badge/Status-Part%201%20Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This project is Part 1 of a Deep Learning in Operations Research (DLOR) coursework. It explores traffic sign recognition using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The notebook performs extensive data exploration, visualization, and analysis, laying the groundwork for future model training using convolutional neural networks (CNNs).

---

## 📘 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Features & Analyses](#features--analyses)
- [Installation](#installation)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Sample Outputs](#sample-outputs)
- [Learnings & Challenges](#learnings--challenges)
- [Next Steps](#next-steps)
- [Model Analysis & Improvement](#-model-analysis--improvement-roadmap)
- [Author](#author)
- [License](#license)

---

## 📌 Overview

Autonomous vehicles and advanced driving systems must detect and understand traffic signs in real time to make informed decisions. This project focuses on preparing, inspecting, and understanding a real-world traffic sign dataset to build a foundation for machine learning models in this domain.

---

## 📂 Dataset

**GTSRB** – German Traffic Sign Recognition Benchmark  
- **43** traffic sign classes  
- **50,000+** real-world traffic images  
- Public benchmark from IJCNN 2011  
- Diverse in resolution, lighting, and angle  

---

## 🎯 Objectives

- Import and explore the GTSRB dataset
- Map label indices to human-readable names
- Visualize and analyze images and class distributions
- Prepare a clean, high-resolution dataset sample
- Identify challenges in preprocessing and modeling

---

## 🔍 Features & Analyses

- ✅ Dataset loading with `deeplake`
- ✅ Grid image visualization per class
- ✅ Class distribution comparison (train/test)
- ✅ Label mapping and feature correlation matrix
- ✅ Image resizing, normalization, and reshaping
- ✅ Duplicate image detection
- ✅ Data augmentation with `ImageDataGenerator`
- ✅ CNN model design and evaluation


---

## 🧪 Notebook Walkthrough

1. **Load Data** – via Deep Lake.
2. **Visual Inspection** – high-res images, missing labels.
3. **Label Mapping** – convert numeric labels to class names.
4. **EDA** – aspect ratios, label counts, correlations.
5. **Preprocessing** – resize, normalize, encode.
6. **Augmentation** – simulate real-world variation.
7. **CNN Model** – conv layers, pooling, dropout, softmax output.

---

## 🖼 Sample Outputs

- ✅ 43-class image grid
- ✅ Label distribution bar charts
- ✅ Aspect ratio histogram
- ✅ Heatmap of feature correlations
- ✅ Augmented image visualizations
- ✅ CNN architecture summary

---

## 🎓 Learnings & Challenges

| Challenge | Solution |
|----------|----------|
| Missing label visual | Explicitly extracted label 39 manually |
| Dataset imbalance | Visualized with bar plots |
| Image variance | Normalized and resized all to 32×32 |
| Redundancy | Detected duplicates using image hashing |

---

## 🚀 Next Steps

- Train with deeper CNNs or transfer learning
- Analyze confusion matrix in more detail
- Tune hyperparameters (LR, batch size, dropout)
- Consider K-fold CV for robust validation
- Deploy in simulated environment (optional)

---

## 🧠 Model Analysis & Improvement Roadmap

### 🔍 What Was Done

#### 🧹 Data Preparation
- Resized all GTSRB images to **32x32 pixels**.
- Normalized pixel values to `[0, 1]`.
- One-hot encoded the labels for multi-class output.

#### 🔁 Data Augmentation
- Used rotation, zoom, brightness, shift, flip, and shear to increase dataset diversity.

#### 🧠 Model Architecture
- Three `Conv2D` + `MaxPooling2D` layers
- One `Dense(128)` + `Dropout(0.5)` + softmax for 43 classes

#### ⚙️ Training
- Adam optimizer with early stopping based on validation accuracy

#### 📈 Evaluation
- Confusion matrix and misclassified image visualization

---

### 🔧 What Could Be Improved

#### 🔄 Enhanced Data Augmentation
- Add perspective transforms and occlusion simulation.

#### 🧠 Transfer Learning with Pre-Trained Models
- Fine-tune ResNet or MobileNet for better feature extraction.

#### ⚖️ Class Weighting
- Balance loss contribution from underrepresented classes.

#### 🔁 K-Fold Cross-Validation
- Evaluate across multiple splits to reduce overfitting bias.

#### 📉 Regularization
- Add L2 penalty to reduce large weight overfitting.

#### 🎛 Hyperparameter Optimization
- Use grid/random search to tune learning rate, batch size, etc.

#### 📊 Learning Rate Scheduling
- Adjust LR dynamically during training for better convergence.

---

### 🧭 Next Phase Action Plan

1. Confusion matrix-driven augmentation
2. Transfer learning with MobileNet or ResNet
3. 5-fold cross-validation setup
4. Integrate LR scheduler (`ReduceLROnPlateau`)

---

### ✅ Conclusion

By implementing these improvements, the model will be more robust and reliable for real-world use in autonomous driving and intelligent transport systems. This roadmap bridges the gap between academic prototyping and production-ready deployment.

---

## 👨‍💻 Author

**Moanish Ashok Kumar**  
Applied AI Student · Computer Vision Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/moanish-ashok-kumar-086978272/)

---
