import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
try:
    from preprocessing import load_preprocessed_data
    SAVE_DIR = r"Celeb-DF\preprocessed"
    _, _, test_loader, _ = load_preprocessed_data(SAVE_DIR)  # Only need test data
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Define ResNet50 Model (same architecture as training)
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = resnet50(weights=None)  # No pre-trained weights
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Binary classification

    def forward(self, x):
        return self.model(x)

# Load trained model
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode
print("✅ Model loaded successfully!")

# **Evaluate Model**
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store actual labels

    return np.array(all_labels), np.array(all_preds)

# Get predictions & true labels
y_true, y_pred = evaluate_model(model, test_loader)

# **Calculate Metrics**
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="binary")
recall = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")

# **Print Metrics**
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# **Generate Confusion Matrix**
cm = confusion_matrix(y_true, y_pred)
labels = ["Real", "Fake"]

# **Plot Confusion Matrix**
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# **Bar Chart for Evaluation Metrics**
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
plt.ylim(0, 1)
plt.title("Model Evaluation Metrics")
plt.show()
