import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


IMG_SIZE = (224, 224)  # Standard input size for CNNs
NUM_FRAMES_PER_VIDEO = 10  # Number of frames to extract per video
BATCH_SIZE = 32

# Function to load and preprocess a single frame (unchanged)
def load_and_preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMG_SIZE)
    frame = frame / 255.0
    return frame

# Function to extract frames from a video (unchanged)
def extract_frames_from_video(video_path, num_frames=NUM_FRAMES_PER_VIDEO):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Could not read frames from {video_path}")
        vidcap.release()
        return frames
    
    step = max(1, total_frames // num_frames) if total_frames > num_frames else 1
    success, frame = vidcap.read()
    count = 0
    extracted = 0
    
    while success and extracted < num_frames:
        if count % step == 0:
            processed_frame = load_and_preprocess_frame(frame)
            frames.append(processed_frame)
            extracted += 1
        success, frame = vidcap.read()
        count += 1
    
    vidcap.release()
    return frames

# Function to load dataset from video directories (unchanged)
def load_dataset_from_videos(real_dir, fake_dir):
    images = []
    labels = []
    
    for filename in os.listdir(real_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(real_dir, filename)
            frames = extract_frames_from_video(video_path)
            if frames:
                images.extend(frames)
                labels.extend([0] * len(frames))
            else:
                print(f"Skipping {filename}: No frames extracted")
    
    for filename in os.listdir(fake_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(fake_dir, filename)
            frames = extract_frames_from_video(video_path)
            if frames:
                images.extend(frames)
                labels.extend([1] * len(frames))
            else:
                print(f"Skipping {filename}: No frames extracted")
    
    images = np.array(images, dtype=np.float32)  # [num_samples, 224, 224, 3]
    labels = np.array(labels, dtype=np.int64)    # [num_samples]
    
    return images, labels

# Custom Dataset class for PyTorch (unchanged)
class DeepfakeDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)  # [N, C, H, W]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to compute class weights
def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)
    weights = n_samples / (n_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

# Main preprocessing function with class weighting
def preprocess_data(real_dir, fake_dir, save_dir=None):
    # Load and preprocess videos
    print("Loading dataset from videos...")
    X, y = load_dataset_from_videos(real_dir, fake_dir)
    print(f"Loaded {len(X)} frames with shape {X.shape}")
    
    if len(X) == 0:
        raise ValueError("No frames were extracted. Check video files or directories.")
    
    # Split the dataset: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Save preprocessed data if save_dir is provided
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "X_train.npy"), X_train)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train)
        np.save(os.path.join(save_dir, "X_val.npy"), X_val)
        np.save(os.path.join(save_dir, "y_val.npy"), y_val)
        np.save(os.path.join(save_dir, "X_test.npy"), X_test)
        np.save(os.path.join(save_dir, "y_test.npy"), y_test)
        print(f"Preprocessed data saved to {save_dir}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Compute class weights for training set
    class_weights = compute_class_weights(y_train)
    sample_weights = [class_weights[label] for label in y_train]
    
    # Create weighted sampler for training
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
    
    # Create PyTorch datasets
    train_dataset = DeepfakeDataset(X_train, y_train, transform=transform)
    val_dataset = DeepfakeDataset(X_val, y_val, transform=transform)
    test_dataset = DeepfakeDataset(X_test, y_test, transform=transform)
    
    # Create DataLoaders (use sampler for training)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_weights  # Return class_weights for use in loss function

# Function to load saved data with class weighting
def load_preprocessed_data(save_dir):
    X_train = np.load(os.path.join(save_dir, "X_train.npy"))
    y_train = np.load(os.path.join(save_dir, "y_train.npy"))
    X_val = np.load(os.path.join(save_dir, "X_val.npy"))
    y_val = np.load(os.path.join(save_dir, "y_val.npy"))
    X_test = np.load(os.path.join(save_dir, "X_test.npy"))
    y_test = np.load(os.path.join(save_dir, "y_test.npy"))
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Compute class weights for training set
    class_weights = compute_class_weights(y_train)
    sample_weights = [class_weights[label] for label in y_train]
    
    # Create weighted sampler for training
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
    
    train_dataset = DeepfakeDataset(X_train, y_train, transform=transform)
    val_dataset = DeepfakeDataset(X_val, y_val, transform=transform)
    test_dataset = DeepfakeDataset(X_test, y_test, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_weights

# Example usage
if __name__ == "__main__":
    # Specify your video directories
    REAL_DIR = r"C:\Parvesh\College\4th Sem\Project\Dataset\Model_data/real"  # Folder with real videos
    FAKE_DIR = r"C:\Parvesh\College\4th Sem\Project\Dataset\Model_data\fake"  # Folder with fake videos
    SAVE_DIR = r"C:\Parvesh\College\4th Sem\Project\Dataset\Model_data\processed"  # Directory to save preprocessed data
    
    # Preprocess the data and save it
    try:
        train_loader, val_loader, test_loader, class_weights = preprocess_data(
            real_dir=REAL_DIR,
            fake_dir=FAKE_DIR,
            save_dir=SAVE_DIR
        )
        
        # Example: Iterate over the training data
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
            break
        
        print(f"Class weights: {class_weights}")
        print("Data preprocessing complete with class weighting. Ready for training with PyTorch!")
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    
    # Optional: Load saved data later
    # train_loader, val_loader, test_loader, class_weights = load_preprocessed_data(SAVE_DIR)