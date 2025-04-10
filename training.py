import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress tracking
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to preprocessed dataset
SAVE_DIR = r"Celeb-DF\preprocessed"

# Load preprocessed dataset and class weights
try:
    from preprocessing import load_preprocessed_data  # Import function from preprocessing script
    train_loader, val_loader, test_loader, class_weights = load_preprocessed_data(SAVE_DIR)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Define ResNet50 model for binary classification
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        weights = ResNet50_Weights.DEFAULT  # Use pretrained weights
        self.model = resnet50(weights=weights)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Binary classification (real vs fake)

    def forward(self, x):
        return self.model(x)

# Initialize model and move to device
model = DeepfakeDetector().to(device)

# Define loss function with class weighting
class_weights = class_weights.to(device) if isinstance(class_weights, torch.Tensor) else torch.tensor([1.0, 1.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Added weight decay for regularization
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    best_val_acc = 0.0  # Track the best validation accuracy

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best model updated at epoch {epoch+1}")

        scheduler.step()  # Reduce learning rate

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10)

# Load best model for evaluation
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Test function
def evaluate_model(model, test_loader):
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"📊 Test Accuracy: {test_acc:.4f}")

# Evaluate the model on the test set
evaluate_model(model, test_loader)
