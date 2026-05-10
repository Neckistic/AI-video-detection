# ==============================================================================
# train_deepfake_f3.py
# Deepfake Detection Model Training Script (稳定版本)
# Backbone: EfficientNet-B4
# Face Detector: MTCNN
# Framework: PyTorch
# 特点: 使用 torchvision 替代 albumentations，彻底避免依赖冲突
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from sklearn.metrics import classification_report

# PyTorch and related libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Computer Vision and Machine Learning libraries
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from facenet_pytorch import MTCNN

# ==============================================================================
# 1. Configuration Parameters (稳定版本)
# ==============================================================================
# --- Paths ---
DATA_DIRS = [
    r"C:\Users\User\Desktop\COMP4471Project\dfdc_train_part_0"

]
    # r"C:\Users\User\Desktop\COMP4471Project\dfdc_train_part_1",
    # r"C:\Users\User\Desktop\COMP4471Project\dfdc_train_part_2"
# --- Model & Training Parameters ---
MODEL_NAME = 'tf_efficientnet_b3_ns'
IMG_SIZE = 300
MAX_SEQ_LENGTH = 8
BATCH_SIZE = 8
EPOCHS = 6
LEARNING_RATE = 1e-4
TEST_SIZE = 0.1
VAL_SIZE = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"=== Deepfake Detection Training (Stable Version) ===")
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"Model: {MODEL_NAME} | Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Sequence length: {MAX_SEQ_LENGTH} | Batch size: {BATCH_SIZE}")
print(f"Test set ratio: {TEST_SIZE*100}% from training data")

# ==============================================================================
# 2. Data Loading and Splitting
# ==============================================================================
def load_and_split_dataset(data_dirs, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=42):
    """
    Load metadata from multiple directories and split into train, validation, and test sets.
    """
    print("\n[Step 1/7] Loading dataset metadata from multiple directories...")
    video_paths = []
    labels = []

    for data_dir in data_dirs:
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Warning: metadata.json not found in {data_dir}, skipping.")
            continue

        print(f"--> Loading from {data_dir}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for video_name, video_info in tqdm(metadata.items(), desc=f"Processing {os.path.basename(data_dir)}"):
            if video_info['label'] in ['FAKE', 'REAL']:
                video_path = os.path.join(data_dir, video_name)
                if os.path.exists(video_path):
                    video_paths.append(video_path)
                    labels.append(1 if video_info['label'] == 'FAKE' else 0)

    if not video_paths:
        raise FileNotFoundError("No video files found. Please check your DATA_DIRS paths.")

    print(f"\nTotal videos loaded: {len(video_paths)}")
    print(f"  - Fake videos: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  - Real videos: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    print("\nSplitting dataset into train, validation, and test sets...")
    
    # First split: Separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        video_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: Split remaining data into train and validation
    val_relative_size = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_relative_size,
        random_state=random_state, stratify=train_val_labels
    )

    print(f"\nDataset split summary:")
    print(f"  - Train set:   {len(train_paths)} videos ({sum(train_labels)} fake, {len(train_labels)-sum(train_labels)} real)")
    print(f"  - Validation set: {len(val_paths)} videos ({sum(val_labels)} fake, {len(val_labels)-sum(val_labels)} real)")
    print(f"  - Test set:       {len(test_paths)} videos ({sum(test_labels)} fake, {len(test_labels)-sum(test_labels)} real)")

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

# ==============================================================================
# 3. Face Detector (MTCNN) - unchanged
# ==============================================================================
class MTCNNFaceDetector:
    """
    Wrapper for facenet-pytorch MTCNN implementation.
    """
    def __init__(self, device):
        print("\n[Step 2/7] Initializing MTCNN face detector...")
        self.device = device
        self.mtcnn = MTCNN(keep_all=False, device=device)
        print("✓ MTCNN detector initialized.")

    def extract_face(self, frame, padding_ratio=0.2):
        """
        Extract a single face from a frame.
        """
        frame_pil = Image.fromarray(frame)
        boxes, _ = self.mtcnn.detect(frame_pil)

        if boxes is None:
            return frame, False

        x1, y1, x2, y2 = boxes[0]
        w, h = x2 - x1, y2 - y1

        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)

        x_start = max(0, int(x1) - pad_x)
        y_start = max(0, int(y1) - pad_y)
        x_end = min(frame.shape[1], int(x2) + pad_x)
        y_end = min(frame.shape[0], int(y2) + pad_y)

        face_roi = frame[y_start:y_end, x_start:x_end]

        if face_roi.size == 0:
            return frame, False

        return face_roi, True

# ==============================================================================
# 4. PyTorch Dataset and Augmentations (使用 torchvision)
# ==============================================================================
def add_gaussian_noise(tensor, mean=0., std=0.01):
    """Add Gaussian noise to tensor"""
    return tensor + torch.randn(tensor.size()) * std + mean

class VideoAugmentation:
    """Video data augmentation pipeline using torchvision"""
    def __init__(self, is_train=True):
        # Base transforms: resize, convert to tensor, normalize
        base_transforms = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ]
        
        if is_train:
            # Augmentation during training
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                *base_transforms  # Unpack base transforms
            ])
            self.add_noise = True  # Flag to add noise
        else:
            # Only base transforms for validation/testing
            self.transform = transforms.Compose(base_transforms)
            self.add_noise = False
    
    def __call__(self, image):
        # Input should be numpy array (H, W, C) RGB format
        image_pil = Image.fromarray(image)
        tensor = self.transform(image_pil)
        
        # Add Gaussian noise if needed
        if self.add_noise and random.random() < 0.3:
            tensor = add_gaussian_noise(tensor, std=0.02)
            
        return tensor

class DeepfakeVideoDataset(Dataset):
    """PyTorch Dataset for loading and processing deepfake videos"""
    def __init__(self, video_paths, labels, face_detector, is_train=True):
        self.video_paths = video_paths
        self.labels = labels
        self.face_detector = face_detector
        self.is_train = is_train
        self.augmentation = VideoAugmentation(is_train=is_train)

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video {video_path}")
            return torch.zeros((MAX_SEQ_LENGTH, 3, IMG_SIZE, IMG_SIZE))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return torch.zeros((MAX_SEQ_LENGTH, 3, IMG_SIZE, IMG_SIZE))

        frame_indices = np.linspace(0, total_frames - 1, MAX_SEQ_LENGTH, dtype=int)
        
        face_detections = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                processed_frame = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract face using MTCNN
                face_roi, success = self.face_detector.extract_face(frame_rgb)
                if success:
                    face_detections += 1
                
                # Apply augmentation (including resize and normalization)
                processed_frame = self.augmentation(face_roi)
            
            frames.append(processed_frame)

        cap.release()
        
        if not frames:
            return torch.zeros((MAX_SEQ_LENGTH, 3, IMG_SIZE, IMG_SIZE))
            
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.extract_frames(video_path)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return frames, label_tensor

# ==============================================================================
# 5. Model Architecture (EfficientNet-B3 + GRU + Attention)
# ==============================================================================
class EfficientNetDeepfakeDetector(nn.Module):
    """EfficientNet-B3 backbone with temporal GRU and attention mechanism"""
    def __init__(self, model_name=MODEL_NAME, pretrained=True):
        super().__init__()
        # 1. Backbone feature extractor (EfficientNet-B3)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        feature_dim = self.backbone.num_features  # For B3, this is 1536
        print(f"✓ Backbone '{model_name}' loaded, feature dimension: {feature_dim}")

        # 2. Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Temporal processor (Bidirectional GRU)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 4. Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # 5. Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        features = self.backbone(x_reshaped)
        pooled_features = self.adaptive_pool(features).squeeze(-1).squeeze(-1)
        sequence_features = pooled_features.view(batch_size, seq_len, -1)
        
        gru_out, _ = self.gru(sequence_features)
        attention_weights = self.attention(gru_out)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        output = self.classifier(context_vector)
        
        return output

# ==============================================================================
# 6. Training and Evaluation Logic - unchanged
# ==============================================================================
class ModelTrainer:
    """Wrapper class for training, validation, and evaluation loops"""
    def __init__(self, model, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.best_val_f1 = 0
        self.history = {k: [] for k in ['train_loss', 'train_acc', 'train_f1', 
                                       'val_loss', 'val_acc', 'val_f1']}

    def _run_epoch(self, dataloader, criterion, optimizer=None):
        is_train = optimizer is not None
        if is_train:
            self.model.train()
            mode = "Training"
        else:
            self.model.eval()
            mode = "Validation"

        total_loss = 0
        all_preds, all_targets = [], []
        
        pbar = tqdm(dataloader, desc=f"{mode} epoch", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device).squeeze()

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, targets)
                
                if is_train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        return epoch_loss, epoch_acc, epoch_f1

    def train(self, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
        print("\n[Step 5/7] Starting model training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            train_loss, train_acc, train_f1 = self._run_epoch(train_loader, criterion, optimizer)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            
            val_loss, val_acc, val_f1 = self._run_epoch(val_loader, criterion)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            if scheduler:
                scheduler.step(val_loss)

            print(f"Train -> Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"Val -> Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                print(f"🎉 New best model found! F1: {val_f1:.4f}. Saving model...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'history': self.history,
                }, 'best_model.pth')
        
        total_training_time = time.time() - start_time
        print(f"\nTraining completed, time: {total_training_time/60:.2f} minutes.")
        print(f"Best validation F1 score: {self.best_val_f1:.4f}")
        
        checkpoint = torch.load('best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.history

    def evaluate(self, test_loader, set_name="Test"):
        """Evaluate model and return comprehensive metrics"""
        print(f"\n[Step 7/7] Evaluating on {set_name} set...")
        self.model.eval()
        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Evaluating {set_name}"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).squeeze()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(float)
                
                all_probs.extend(np.atleast_1d(probs))
                all_preds.extend(np.atleast_1d(preds))
                all_targets.extend(targets.numpy())

        results = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
        }
        
        print(f"\n--- {set_name} Set Performance ---")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Precision: {results['precision']:.4f}")
        print(f"  - Recall: {results['recall']:.4f}")
        print(f"  - F1 Score: {results['f1']:.4f}")
        print(f"  - ROC-AUC: {results['roc_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(all_targets, all_preds, target_names=['Real', 'Fake']))
        
        return results

# ==============================================================================
# 7. Visualization Functions - Updated with English labels
# ==============================================================================
def plot_training_history(history):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle("Training History and Metrics", fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss', lw=2, color='blue', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', lw=2, color='red', alpha=0.8)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend()
    
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', lw=2, color='blue', alpha=0.8)
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', lw=2, color='red', alpha=0.8)
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()
    
    axes[1, 0].plot(history['train_f1'], label='Train F1-Score', lw=2, color='blue', alpha=0.8)
    axes[1, 0].plot(history['val_f1'], label='Val F1-Score', lw=2, color='red', alpha=0.8)
    axes[1, 0].set_title('Training and Validation F1-Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend()
    
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(results, set_name="Test"):
    """Plot annotated confusion matrix"""
    cm = confusion_matrix(results['targets'], results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'],
                annot_kws={"size": 16})
    
    plt.title(f'Confusion Matrix ({set_name} Set)\nAccuracy: {results["accuracy"]:.4f}, F1-Score: {results["f1"]:.4f}', 
              fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(results, set_name="Test"):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(results['targets'], results['probabilities'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve ({set_name} Set)', fontsize=14, pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'roc_curve_{set_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_bar_chart(train_results, val_results, test_results):
    """Plot metrics comparison bar chart across different sets"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    train_scores = [train_results[m] for m in metrics]
    val_scores = [val_results[m] for m in metrics]
    test_scores = [test_results[m] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    plt.bar(x - width, train_scores, width, label='Train', color='blue', alpha=0.7)
    plt.bar(x, val_scores, width, label='Validation', color='green', alpha=0.7)
    plt.bar(x + width, test_scores, width, label='Test', color='red', alpha=0.7)
    
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Comparison Across Sets', fontsize=14, pad=20)
    plt.xticks(x, labels)
    plt.ylim([0, 1.0])
    plt.legend(loc='lower right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add value annotations on top of bars
    for i, (train_val, val_val, test_val) in enumerate(zip(train_scores, val_scores, test_scores)):
        plt.text(i - width, train_val + 0.01, f'{train_val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i, val_val + 0.01, f'{val_val:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width, test_val + 0.01, f'{test_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_csv(train_results, val_results, test_results):
    """Save all results to CSV file for future reference"""
    results_dict = {
        'Set': ['Train', 'Validation', 'Test'],
        'Accuracy': [train_results['accuracy'], val_results['accuracy'], test_results['accuracy']],
        'Precision': [train_results['precision'], val_results['precision'], test_results['precision']],
        'Recall': [train_results['recall'], val_results['recall'], test_results['recall']],
        'F1-Score': [train_results['f1'], val_results['f1'], test_results['f1']],
        'ROC-AUC': [train_results['roc_auc'], val_results['roc_auc'], test_results['roc_auc']],
    }
    
    df = pd.DataFrame(results_dict)
    df.to_csv('model_performance_results.csv', index=False)
    print("\n✓ Results saved to 'model_performance_results.csv'")
    
    with open('detailed_classification_reports.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("Training Set Classification Report\n")
        f.write("="*60 + "\n")
        f.write(classification_report(train_results['targets'], train_results['predictions'], 
                                     target_names=['Real', 'Fake']))
        
        f.write("\n" + "="*60 + "\n")
        f.write("Validation Set Classification Report\n")
        f.write("="*60 + "\n")
        f.write(classification_report(val_results['targets'], val_results['predictions'], 
                                     target_names=['Real', 'Fake']))
        
        f.write("\n" + "="*60 + "\n")
        f.write("Test Set Classification Report\n")
        f.write("="*60 + "\n")
        f.write(classification_report(test_results['targets'], test_results['predictions'], 
                                     target_names=['Real', 'Fake']))

# ==============================================================================
# Main Execution Block
# ==============================================================================
def main():
    # 1. Load and split data
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
        load_and_split_dataset(DATA_DIRS)

    # 2. Initialize face detector
    face_detector = MTCNNFaceDetector(device=DEVICE)

    # 3. Create datasets and data loaders
    print("\n[Step 3/7] Creating PyTorch datasets and data loaders...")
    train_dataset = DeepfakeVideoDataset(train_paths, train_labels, face_detector, is_train=True)
    val_dataset = DeepfakeVideoDataset(val_paths, val_labels, face_detector, is_train=False)
    test_dataset = DeepfakeVideoDataset(test_paths, test_labels, face_detector, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"✓ Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # 4. Create model and training components
    print("\n[Step 4/7] Initializing model and optimizer...")
    model = EfficientNetDeepfakeDetector()
    
    # Calculate class weights for imbalance handling
    fake_count = sum(train_labels)
    real_count = len(train_labels) - fake_count
    pos_weight = torch.tensor([real_count / fake_count], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using weighted loss. Fake class weight: {pos_weight.item():.2f} (fake: {fake_count}, real: {real_count})")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # 5. Train model
    trainer = ModelTrainer(model, device=DEVICE)
    history = trainer.train(train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS)

    # 6. Evaluate on all sets
    print("\n[Step 6/7] Evaluating model on all sets...")
    
    # Evaluate on train set (for reference)
    train_results = trainer.evaluate(train_loader, set_name="Train")
    
    # Evaluate on validation set
    val_results = trainer.evaluate(val_loader, set_name="Validation")
    
    # Evaluate on test set (10% from training data)
    test_results = trainer.evaluate(test_loader, set_name="Test")

    # 7. Generate comprehensive visualizations
    print("\n[Step 7/7] Generating visualizations...")
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrices for all sets
    plot_confusion_matrix(train_results, set_name="Train")
    plot_confusion_matrix(val_results, set_name="Validation")
    plot_confusion_matrix(test_results, set_name="Test")
    
    # Plot ROC curves for all sets
    plot_roc_curve(train_results, set_name="Train")
    plot_roc_curve(val_results, set_name="Validation")
    plot_roc_curve(test_results, set_name="Test")
    
    # Plot comparison bar chart
    plot_metrics_bar_chart(train_results, val_results, test_results)
    
    # Save results to CSV
    save_results_to_csv(train_results, val_results, test_results)
    
    print("\n" + "="*60)
    print("=== Training Completed ===")
    print("="*60)
    print("\nModel Performance Summary:")
    print(f"  - Best Validation F1: {trainer.best_val_f1:.4f}")
    print(f"  - Test Set Accuracy: {test_results['accuracy']:.4f}")
    print(f"  - Test Set F1-Score: {test_results['f1']:.4f}")
    print(f"  - Test Set ROC-AUC: {test_results['roc_auc']:.4f}")
    
    print("\nGenerated Files:")
    print("  - best_model.pth (best model weights)")
    print("  - training_history.png (training metrics plot)")
    print("  - confusion_matrix_*.png (confusion matrices)")
    print("  - roc_curve_*.png (ROC curves)")
    print("  - performance_comparison.png (performance comparison)")
    print("  - model_performance_results.csv (detailed results)")
    print("  - detailed_classification_reports.txt (classification reports)")
    print("\n✓ All done! You can now analyze model performance.")

if __name__ == "__main__":
    main()