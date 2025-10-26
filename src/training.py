import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import json
from typing import List, Tuple, Dict
import random

class ViolenceDataset(Dataset):
    """Custom dataset for violence detection from video frames"""
    
    def __init__(self, frame_paths: List[str], labels: List[int], transform=None):
        self.frame_paths = frame_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        # load frame
        frame_path = self.frame_paths[idx]
        try:
            image = Image.open(frame_path).convert('RGB')
        except:
            # create dummy image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

class ViolenceTrainer:
    """Trainer class for the violence detection CNN"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self, dataloader):
        """train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=20):
        """train the model for multiple epochs"""
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 20)
            
            # train
            train_loss, train_acc = self.train_epoch(train_loader)
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # validate
            val_loss, val_acc = self.validate(val_loader)
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'violence_model_best.pth')
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
            
            self.scheduler.step()
        
        return best_val_acc

def extract_frames_for_training(data_dir: str, max_frames_per_video: int = 10) -> Tuple[List[str], List[int]]:
    """extract frames from videos for training"""
    frame_paths = []
    labels = []
    
    # create frames directory
    frames_dir = os.path.join(data_dir, 'training_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # process each category
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) 
                  and d not in ['Labels', 'DCSASS Dataset', 'training_frames']]
    
    print(f"Found {len(categories)} categories: {categories}")
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        video_dirs = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        
        print(f"Processing {category} with {len(video_dirs)} videos...")
        
        for video_dir in video_dirs[:20]:  # limit to first 20 videos per category
            video_path = os.path.join(category_path, video_dir)
            
            # determine label (0 = normal, 1 = abnormal)
            label = 1 if category in ['Fighting', 'Robbery', 'Assault', 'Arson', 'Shooting'] else 0
            
            # extract frames
            cap = cv2.VideoCapture(os.path.join(video_path, os.listdir(video_path)[0]))
            frame_count = 0
            
            while frame_count < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # save frame
                frame_filename = f"{category}_{video_dir}_frame_{frame_count}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                frame_paths.append(frame_path)
                labels.append(label)
                frame_count += 1
            
            cap.release()
    
    print(f"Extracted {len(frame_paths)} frames total")
    print(f"Normal frames: {labels.count(0)}, Abnormal frames: {labels.count(1)}")
    
    return frame_paths, labels

def train_violence_model(data_dir: str = "./data", epochs: int = 10):
    """main training function"""
    print("🚀 Starting violence detection model training...")
    
    # check if we have extracted frames
    frames_dir = os.path.join(data_dir, 'training_frames')
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        print("📸 Extracting frames from videos...")
        frame_paths, labels = extract_frames_for_training(data_dir)
    else:
        print("📸 Loading existing frames...")
        frame_paths = []
        labels = []
        
        for filename in os.listdir(frames_dir):
            if filename.endswith('.jpg'):
                frame_paths.append(os.path.join(frames_dir, filename))
                # determine label from filename
                if any(abnormal in filename for abnormal in ['Fighting', 'Robbery', 'Assault', 'Arson', 'Shooting']):
                    labels.append(1)
                else:
                    labels.append(0)
    
    if len(frame_paths) == 0:
        print("❌ No frames found! Please check your data directory.")
        return None
    
    # split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        frame_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📊 Training samples: {len(train_paths)}")
    print(f"📊 Validation samples: {len(val_paths)}")
    
    # data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # create datasets
    train_dataset = ViolenceDataset(train_paths, train_labels, train_transform)
    val_dataset = ViolenceDataset(val_paths, val_labels, val_transform)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # create model
    model = ViolenceCNN(num_classes=2)
    print(f"📱 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # train model
    trainer = ViolenceTrainer(model, device)
    best_acc = trainer.train(train_loader, val_loader, epochs)
    
    print(f"🎯 Training completed! Best validation accuracy: {best_acc:.2f}%")
    
    return model

if __name__ == "__main__":
    # train the model
    model = train_violence_model(epochs=5)  # reduced epochs for demo