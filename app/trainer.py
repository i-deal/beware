from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from app.optimized_data_loader import OptimizedDCSASSDataLoader as DCSASSDataLoader
import wandb
from tqdm import tqdm
import os
from datetime import datetime
from app.cnn_model import load_model


if __name__ == "__main__":
    # train the model
    model = load_model() #ViolenceCNN(2)
    device = 'cpu'
    if not os.path.exists('checkpoints/run1/'):
        os.mkdir('checkpoints/run1/')
    #checkpoint = torch.load(f'checkpoints/run1/checkpoint.pth', device, weights_only = True)
    #model.load_state_dict(checkpoint['state_dict'])
    print('load data')
    repo_root = Path(__file__).parent.parent  # backend/
    data_root = repo_root / "data" / "DCSASS Dataset"
    dataloader = DCSASSDataLoader()
    train_loader = dataloader.train_loader
    test_loader = dataloader.test_loader
    print('start train')
    model.train_full(train_loader, test_loader, device, 'run1')