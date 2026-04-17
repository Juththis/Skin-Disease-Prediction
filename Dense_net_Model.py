# train.py

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader
from collections import Counter

# Preprocessing

def preprocess_image(image):
    image = np.array(image)

    cropped = image[25:425, 25:575]
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


# Handle Class Imbalance

class_counts = Counter(train_data.targets)
counts = np.array([class_counts[i] for i in range(num_classes)])

# inverse frequency
class_weights = 1.0 / counts
class_weights = class_weights / class_weights.sum()  # normalize

class_weights = torch.tensor(class_weights, dtype=torch.float32)


# Model

model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Training setup
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training loop
epochs = 15

for epoch in range(epochs):

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


# Save model

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_data.classes
}, "skin_model.pth")

print("Model saved as skin_model.pth")