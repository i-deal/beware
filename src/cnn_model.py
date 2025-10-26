import torch
import torch.nn as nn
import torch.optim as optim
#from dcsass_data_loader import DCSASSDataLoader
from optimized_data_loader import OptimizedDCSASSDataLoader as DCSASSDataLoader
import wandb
from wandb_setup import initialize_wandb, log_system_metrics
from pathlib import Path
from tqdm import tqdm


labels = {'Assault':0, 'Robbery':1, 'Shoplifting':2, 'Shooting':3, 'Normal':4}
reverse_labels = {3:'Assault', 8:'Robbery', 10:'Shoplifting', 9:'Shooting', 11:'Stealing'} # else: 'Normal'

crime_to_label = {
            'Abuse': 0,
            'Arrest': 1, 
            'Arson': 2,
            'Assault': 3,
            'Burglary': 4,
            'Explosion': 5,
            'Fighting': 6,
            'RoadAccidents': 7,
            'Robbery': 8,
            'Shooting': 9,
            'Shoplifting': 10,
            'Stealing': 11,
            'Vandalism': 12
        }

class ViolenceCNN(nn.Module):
    
    def __init__(self, num_classes=2):
        super(ViolenceCNN, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.pre_recurrence = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        self.recurrence = nn.Sequential(
            nn.Linear(512, 340),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(340, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, batch):
        # batch: [T, C, H, W]
        y = torch.zeros(1, 256, device=batch.device)
        outputs = []

        for i in range(batch.size(0)):
            x = batch[i].unsqueeze(0)         # [1, 3, 244, 244]
            x = self.features(x)              # [1, 256, 15, 15]
            x = self.pre_recurrence(x)        # [1, 512]
            z = torch.cat([y, x], dim=1)      # [1, 1024]
            y = self.recurrence(z)            # [1, 512]
            outputs.append(y)

        x = self.classifier(y)
        return x
    
    def classify(self, batch:torch.tensor) -> bool:
        x = self.forward(batch)
        x = x.item()
        return x > 0.5

    def train_ep(self, dataloader, optimizer):
        self.train()
        total_loss = 0
        correct = 0
        total = 0
        max_iter = 20
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Train", unit="batch", total=max_iter)):
            data, target = data.to(device), target.to(device)
            if data.dim() == 5:
                data = data.squeeze(0)
            if target.dim() > 1:
                target = target.view(-1)
            optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            if batch_idx >= max_iter:
                break
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy
    
    def test(self, dataloader):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        max_iter = 7
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Test", unit="batch", total=max_iter)):
            data, target = data.to(device), target.to(device)
            if data.dim() == 5:
                data = data.squeeze(0)
            if target.dim() > 1:
                target = target.view(-1)
            output = self.forward(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            if batch_idx >= max_iter:
                break
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy


    def train_full(self, train_dataloader, test_dataloader, device, checkpoint_folder):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        initialize_wandb('crime-detect', {'version':'CNN-RECURSIVE'})

        for epoch in range(0,100):
            train_loss, acc = self.train_ep(train_dataloader, optimizer)
            test_loss, test_acc = self.test(test_dataloader)

            metrics = {'train_loss':train_loss,
                       'train_accuracy': acc,
                       'test_loss':test_loss,
                       'test_acc':test_acc}
            wandb.log(metrics)
            scheduler.step()

            print(metrics)

            checkpoint =  {'state_dict': self.state_dict()}
            if epoch % 4 == 0:
                torch.save(checkpoint, f'checkpoints/{checkpoint_folder}/mVAE_checkpoint.pth')


if __name__ == "__main__":
    # train the model
    model = ViolenceCNN(2)
    device = 'cpu'
    #checkpoint = torch.load(f'checkpoints/run1/mVAE_checkpoint.pth', device, weights_only = True)
    #model.load_state_dict(checkpoint['state_dict'])
    print('load data')
    repo_root = Path(__file__).parent.parent  # backend/
    data_root = repo_root / "data" / "DCSASS Dataset"
    dataloader = DCSASSDataLoader()
    train_loader = dataloader.train_loader
    test_loader = dataloader.test_loader
    print('start train')
    model.train_full(train_loader, test_loader, device, 'run1')
