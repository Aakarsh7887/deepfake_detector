import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm
import os

# **Set Device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **Define Model (Same as Training)**
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = resnet50(weights=None)  # No pre-trained weights
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Binary classification (Real vs Fake)

    def forward(self, x):
        return self.model(x)

# **Load Trained Model**
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# **Preprocessing Function**
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # ResNet50 requires 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    """ Convert frame to tensor for model input """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return transform(frame).unsqueeze(0).to(device)  # Add batch dimension

# **Video Prediction Function with Confidence Score**
def predict_video(video_path, sample_rate=10):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, None

    frame_probs = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_rate == 0:  # Process every nth frame
            frame_tensor = preprocess_frame(frame)
            with torch.no_grad():
                output = model(frame_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
                frame_probs.append(probs.cpu().numpy())

    cap.release()

    if not frame_probs:
        print("No frames processed. Try a different sample rate.")
        return None, None

    # **Average the probabilities across frames**
    avg_probs = np.mean(frame_probs, axis=0)  # Compute mean probability for Real & Fake
    real_prob, fake_prob = avg_probs[0]  # Extract probabilities

    # **Final Prediction**
    final_prediction = "Fake" if fake_prob > real_prob else "Real"
    confidence = max(real_prob, fake_prob) * 100  # Convert to percentage
    
    print(f"Video Prediction: {final_prediction} (Confidence: {confidence:.2f}%)")
    return final_prediction, confidence

# **Run on a Sample Video**
video_path = r"archive\manip\01_03__hugging_happy__ISF9SP4G.mp4"  # Change to your video file
predict_video(video_path)
