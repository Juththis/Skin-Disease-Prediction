import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from tkinter import Tk, filedialog


class_full_names = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}



# File selection

Tk().withdraw()
img_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
)

if not img_path:
    raise ValueError("No image selected")

print("Selected:", img_path)


# Preprocessing

def preprocess_image(image):
    resized = cv2.resize(image, (224, 224))
    cropped = resized[24:200, 24:224]
    resized = cv2.resize(cropped, (224, 224))


    normalized = resized / 255.0
    normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return normalized.astype(np.float32)


# Load model

checkpoint = torch.load("skin_model.pth", map_location="cpu", weights_only=True)

classes = checkpoint["classes"]
num_classes = len(classes)

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()



image = cv2.imread(img_path)

if image is None:
    raise ValueError("Failed to load image")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess_image(image)

image = np.transpose(image, (2, 0, 1))
image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)


with torch.no_grad():
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)
    percentages = probs * 100


# Output 

print("\nPrediction probabilities:\n")

results = list(zip(classes, percentages[0].tolist()))
results.sort(key=lambda x: x[1], reverse=True)

for cls, prob in results:
    full_name = class_full_names.get(cls, cls)
    print(f"{full_name}: {prob:.2f}%")

# Final prediction
top_class = results[0][0]
top_full = class_full_names.get(top_class, top_class)

print(f"\nFinal Prediction: {top_full}")