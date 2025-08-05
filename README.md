# 👁️ Eye Disease Detection using OCT Images

This project focuses on classifying retinal diseases using Optical Coherence Tomography (OCT) images. Leveraging deep learning, the model aims to predict eye conditions like CNV, DME, DRUSEN, and NORMAL from labeled image data.

---

## 📂 Dataset

- **Source**: [Kaggle - Labeled OCT Images](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct)
- **Categories**:
  - `0`: CNV (Choroidal Neovascularization)
  - `1`: DME (Diabetic Macular Edema)
  - `2`: DRUSEN (Drusen)
  - `3`: NORMAL

Each image is a 2D OCT scan labeled with the corresponding diagnosis.

---

## 📊 Model Performance

### ✅ Test Accuracy
- **Accuracy**: `96.80%`
- **Loss**: `0.1312`
- **F1 Score**: `0.6342` *(macro-averaged)*

### 📄 Classification Report

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| CNV (0)    | 0.98      | 0.97   | 0.98     | 3746    |
| DME (1)    | 0.96      | 0.94   | 0.95     | 1161    |
| DRUSEN (2) | 0.90      | 0.85   | 0.88     | 887     |
| NORMAL (3) | 0.98      | 0.99   | 0.99     | 5139    |
| **Overall**|           |        |          | **10933** |

- **Macro Avg**: Precision: `0.95`, Recall: `0.94`, F1-score: `0.95`
- **Weighted Avg**: Precision: `0.97`, Recall: `0.97`, F1-score: `0.97`

---

### 🔢 Confusion Matrix

[[3643 24 71 8]
[ 13 1095 1 52]
[ 60 8 758 61]
[ 3 18 13 5105]]


---

## 🧠 Model Architecture

- **Preprocessing**:
  - Resized images to consistent dimensions
  - Normalization and augmentation (if applied)

- **Model**:
  - CNN architecture (custom or pretrained backbone like ResNet/VGG)
  - Trained with categorical cross-entropy loss
  - Optimized with Adam optimizer and learning rate scheduler

---


---

## 🛠️ Technologies Used

- Python, TensorFlow/Keras or PyTorch
- NumPy, OpenCV, Matplotlib
- Scikit-learn for evaluation

---

## 🚀 How to Run

```bash
git clone https://github.com/BrijeshRakhasiya/Eye-Disease-Prediction.git
cd eye-disease-prediction
jupyter notebook eye-disease-prediction.ipynb
```

# 📌 Future Work
Integration into an assistive diagnosis tool for ophthalmologists

Add Grad-CAM visualizations for explainability

Deploy model via Streamlit or Flask web app

# 📈 Key Learnings
Applied image classification techniques on medical datasets

Gained experience handling imbalanced multi-class data

Improved evaluation with precision, recall, f1-score, and confusion matrix
# 🙋‍♂️ Author
Brijesh Rakhasiya
AI/ML Enthusiast | Data Scientist | Problem Solver


## 📄 License

This project is licensed under the MIT License.

---
**Made ❤️ by Brijesh Rakhasiya**
