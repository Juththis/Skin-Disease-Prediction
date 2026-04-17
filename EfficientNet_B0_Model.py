import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize


# Preprocessing

def preprocess_image(image):
    image = np.array(image)

    resized = cv2.resize(image, (224, 224))
    cropped = resized[24:200, 24:224]
    resized = cv2.resize(cropped, (224, 224))

    normalized = resized / 255.0
    normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return normalized.astype(np.float32)

transform = transforms.Compose([
    transforms.Lambda(preprocess_image),
    transforms.ToTensor()
])


# Load Dataset

train_data = datasets.ImageFolder("train", transform=transform)
val_data = datasets.ImageFolder("val", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)


# Class Imbalance Handling

class_counts = Counter(train_data.targets)
counts = np.array([class_counts[i] for i in range(num_classes)])

class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float32)


# Model

model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Training 
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

for epoch in range(epochs):

    # TRAIN
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}%")


#EVALUATION 

model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())


print("\n=== Classification Report ===\n")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

print("\n=== Confusion Matrix ===\n")
print(confusion_matrix(all_labels, all_preds))

# ROC-AUC (multi-class)
y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))
roc_auc = roc_auc_score(y_true_bin, np.array(all_probs), multi_class='ovr')

print(f"\nROC-AUC (OvR): {roc_auc:.4f}")


#Top-3 Accuracy

top3_correct = 0
total = 0

for probs, label in zip(all_probs, all_labels):
    top3 = np.argsort(probs)[-3:]
    if label in top3:
        top3_correct += 1
    total += 1

top3_acc = 100 * top3_correct / total
print(f"Top-3 Accuracy: {top3_acc:.2f}%")

# Save Model
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_data.classes
}, "skin_model.pth")

print("\nModel saved as skin_model.pth")