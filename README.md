#  Skin Disease Classification using Deep Learning

This project implements a **multi-class skin disease classification system** using deep learning models such as **EfficientNet-B0** and **DenseNet121**. It predicts skin conditions from dermatoscopic images and outputs **probability scores (%)** for each disease.

---

##  Project Structure

```
Skin-Disease-Prediction/
│
├── Data_Classification.py        # Dataset organization / preprocessing
├── Preprocessing.py              # Image preprocessing functions
│
├── Dense_net_Model.py            # DenseNet121 training script
├── EfficientNet_B0_Model.py      # EfficientNet-B0 training script
│
├── Inference_Densenet.py         # DenseNet inference
├── Inference_Efficient_net.py    # EfficientNet inference (with file picker)
│
├── HAM10000_metadata.xlsx       # Dataset metadata
├── skin_model.pth                # Trained model weights
│
└── README.md
```

---

##  Pre Trained Dataset

* Dataset: **HAM10000 (Human Against Machine with 10000 training images)**
* 7 skin disease classes:

| Label | Full Name                                       |
| ----- | ----------------------------------------------- |
| akiec | Actinic Keratoses and Intraepithelial Carcinoma |
| bcc   | Basal Cell Carcinoma                            |
| bkl   | Benign Keratosis-like Lesions                   |
| df    | Dermatofibroma                                  |
| mel   | Melanoma                                        |
| nv    | Melanocytic Nevi                                |
| vasc  | Vascular Lesions                                |

---

##  Features

*  Multi-class classification (7 classes)
*  Pretrained CNN models (EfficientNet, DenseNet)
*  Probability output (% confidence)
*  Class imbalance handling
*  Evaluation metrics:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix
  * ROC-AUC
  * Top-3 Accuracy
*  GUI-based image selection (file picker)

---

##  Preprocessing Pipeline

* Image resizing → `224x224`
* Cropping (region of interest)
* Normalization using ImageNet statistics:

  ```
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
  ```

---

##  Model Architecture

### EfficientNet-B0 (Primary Model)

* Pretrained on ImageNet
* Modified final classification layer (7 classes)
* Better performance for medical imaging

### DenseNet121 (Alternative Model)

* Used for comparison
* Strong feature reuse capability

---

##  Training

Run EfficientNet training:

```bash
python EfficientNet_B0_Model.py
```
## ⚠️ Important Notes

* The path of the dataset should be inclueded in EfficientNet_B0_Model.py before run the code.

Key configurations:

* Loss: CrossEntropyLoss (with class weights)
* Optimizer: Adam
* Learning Rate: 1e-4
* Epochs: 10+

---

##  Inference

Run inference with file picker:

```bash
python Inference_Efficient_net.py
```

### Output Example:

```
Melanoma: 82.34%
Basal Cell Carcinoma: 10.12%
Melanocytic Nevi: 5.21%

Final Prediction: Melanoma
```

---

##  Evaluation Metrics

After training, the model evaluates using:

* Classification Report
* Confusion Matrix
* ROC-AUC (multi-class)
* Top-3 Accuracy

---

## ⚠️ Important Notes

* Dataset should be contain approximately equal data for each   category for more accuracy (If you train with new dataset)
* Class weights are used to reduce bias
* Predictions represent **model confidence**, not clinical diagnosis

---

##  Requirements

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy scikit-learn
```

---

##  Future Improvements

*  Data augmentation for minority classes
*  External validation dataset

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not intended for medical diagnosis**. Always consult a qualified healthcare professional.

---

##  Author

Developed as part of a deep learning project for skin disease classification.

---
