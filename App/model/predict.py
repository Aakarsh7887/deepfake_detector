import torch
import cv2
import numpy as np
import os
import torch.nn.functional as F

from .registry import MODEL_REGISTRY
from .m_def import MyModel

#  GPU Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

#  Load models ONCE (not every request)
loaded_models = {}

def load_models(frame_count):
    global loaded_models

    if frame_count in loaded_models:
        return

    loaded_models[frame_count] = []

    weights_dir = os.path.join(os.path.dirname(__file__), "weights")

    for file in MODEL_REGISTRY[frame_count]:
        path = os.path.join(weights_dir, file)

        model = MyModel()
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()

        loaded_models[frame_count].append((file, model))



#  Preprocessing
def preprocess(video_path, frame_count):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Avoid division by zero
    if total_frames == 0:
        cap.release()
        return None

    step = max(total_frames // frame_count, 1)

    frames = []
    count = 0
    frame_id = 0

    while cap.isOpened() and count < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (112, 112))
        frame = frame / 255.0

        frames.append(frame)

        frame_id += step
        count += 1

    cap.release()

    frames = np.array(frames)

    frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
    frames = frames.unsqueeze(0)
    frames = frames.to(DEVICE)

    return frames

#  Prediction 
def predict_video(video_path, frame_count):
    load_models(frame_count)
    frames = preprocess(video_path, frame_count)

    model_outputs = {}
    fake_scores = []

    with torch.no_grad():
        for name, model in loaded_models[frame_count]:

            output = model(frames)

            
            probs = F.softmax(output, dim=1)

            fake_score = probs[:, 0].item()
            real_score = probs[:, 1].item()

            label = "FAKE" if fake_score > real_score else "REAL"

            confidence = max(fake_score, real_score) * 100

            model_outputs[name] = {
                "label": label,
                "confidence": round(confidence, 2)
            }

            fake_scores.append(fake_score)

    #  
    avg_fake = sum(fake_scores) / len(fake_scores)

    final_label = "FAKE" if avg_fake > 0.5 else "REAL"

    return {
        "models": model_outputs,
        "final": final_label,
        "confidence": round(avg_fake * 100, 2)
    }