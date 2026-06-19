import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

from gender_model import DenseNet161_GGA_Binary, VGG19_GGA_Binary,ResNet101_GGA_Binary,DenseNet161_GGA,DesnseNet161, ResNet101
import numpy as np



def load_model_weights(model, checkpoint_path, device):
    """
    Loads the saved state dictionary into the model.
    """
    checkpoint                          = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} (Epoch: {checkpoint['epoch']})")
    return model

def evaluate_gender_metrics(model, loader, device):
    model.eval()
    all_preds                           = []
    all_labels                          = []

    with torch.no_grad():
        for images, labels in loader:
            images                      = images.to(device)
            outputs                     = model(images)

            ### Convert probabilities to binary predictions (0 or 1) ###
            ### Based on our mapping: 0 = Male, 1 = Female
            preds                       = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds                           = np.array(all_preds).flatten()
    all_labels                          = np.array(all_labels).flatten()

    ### --- Case 1: 'F' is Positive (Label 1), 'M' is Negative (Label 0) --- ###
    precision_f, recall_f, f1_f,_       = precision_recall_fscore_support(
                                            all_labels, all_preds, pos_label=1, average='binary'
    )

    ### --- Case 2: 'M' is Positive (Label 0), 'F' is Negative (Label 1) --- ###
    precision_m, recall_m, f1_m, _      = precision_recall_fscore_support(
        all_labels, all_preds, pos_label=0, average='binary'
    )

    print("-" *30)
    print("CASE 1: Female (F) as Positive")
    print(f"Precision: {precision_f:.4f} | Recall: {recall_f:.4f} | F1-Score: {f1_f:.4f}")


    print("-" * 30)
    print("CASE 2: Male (M) as Positive")
    print(f"Precision: {precision_m:.4f} | Recall: {recall_m:.4f} | F1-Score: {f1_m:.4f}")

    return {
        'F_metrics': (precision_f, recall_f, f1_f),
        'M_metrics': (precision_m, recall_m, f1_m)
    }


def evaluate_gender_metrics_all(model, tr_loader,ts_loader, device):
    model.eval()
    all_preds                           = []
    all_labels                          = []

    with torch.no_grad():
        for images, labels in tr_loader:
            images                      = images.to(device)
            outputs                     = model(images)

            ### Convert probabilities to binary predictions (0 or 1) ###
            ### Based on our mapping: 0 = Male, 1 = Female
            preds                       = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        
        for images, labels in ts_loader:
            images                      = images.to(device)
            outputs                     = model(images)

            ### Convert probabilities to binary predictions (0 or 1) ###
            ### Based on our mapping: 0 = Male, 1 = Female
            preds                       = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds                           = np.array(all_preds).flatten()
    all_labels                          = np.array(all_labels).flatten()

    ### --- Case 1: 'F' is Positive (Label 1), 'M' is Negative (Label 0) --- ###
    precision_f, recall_f, f1_f,_       = precision_recall_fscore_support(
                                            all_labels, all_preds, pos_label=1, average='binary'
    )

    ### --- Case 2: 'M' is Positive (Label 0), 'F' is Negative (Label 1) --- ###
    precision_m, recall_m, f1_m, _      = precision_recall_fscore_support(
        all_labels, all_preds, pos_label=0, average='binary'
    )

    print("-" *30)
    print("CASE 1: Female (F) as Positive")
    print(f"Precision: {precision_f:.4f} | Recall: {recall_f:.4f} | F1-Score: {f1_f:.4f}")


    print("-" * 30)
    print("CASE 2: Male (M) as Positive")
    print(f"Precision: {precision_m:.4f} | Recall: {recall_m:.4f} | F1-Score: {f1_m:.4f}")

    return {
        'F_metrics': (precision_f, recall_f, f1_f),
        'M_metrics': (precision_m, recall_m, f1_m)
    }



# --- 2. Custom Dataset Loader ---
class PalmVeinDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        self.data                       = pd.read_csv(csv_file)
        self.transform                  = transform
        ### Map 'M' to 0 and 'F' to 1 ####
        self.label_dict                 = {'M': 0.0, 'F': 1.0}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path                        = self.data.iloc[idx, 1] # Column: img_dir
        gender_str                      = self.data.iloc[idx, 2] # Column: gender

        image                           = Image.open(img_path).convert('RGB')
        label                           = torch.tensor([self.label_dict[gender_str]], dtype=torch.float32)
        if self.transform:
            image                       = self.transform(image)
        
        return image, label

# --- 1. Training Function ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss                        = 0.0
    correct                             = 0
    total                               = 0
    
    for images, labels in loader:
        images, labels                  = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs                         = model(images)
        
        
        loss                            = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        
        
        running_loss                    += loss.item() * images.size(0)
        
        # Calculate accuracy
        predicted                       = (outputs > 0.5).float()
        total                           += labels.size(0)
        correct                         += (predicted == labels).sum().item()

        
    epoch_loss                          = running_loss / len(loader.dataset)
    epoch_acc                           = 100. * correct / total
    return epoch_loss, epoch_acc

# --- 2. Validation Function ---
def validate(model, loader, criterion, device):
    model.eval()
    running_loss                        = 0.0
    correct                             = 0
    total                               = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels              = images.to(device), labels.to(device)
            
            outputs                     = model(images)
            loss                        = criterion(outputs, labels)
            
            running_loss                += loss.item() * images.size(0)
            
            predicted                   = (outputs > 0.5).float()
            total                       += labels.size(0)
            correct                     += (predicted == labels).sum().item()

            
    val_loss                            = running_loss / len(loader.dataset)
    val_acc                             = 100. * correct / total
    return val_loss, val_acc


# --- 3. Main Execution Loop ---
def run_training(model, train_loader, val_loader, epochs=20, save_path='best_gga_model.pth'):
    device                              = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion                           = nn.BCELoss()
    # optimizer                           = optim.Adam(model.parameters(), lr=1e-4)
    optimizer                           = optim.SGD(model.parameters(), lr=0.01)
    
    # best_val_loss                       = float('inf') # Initialize with infinity
    best_val_acc                        =0.0
    
    print(f"Starting Training on {device}...")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc           = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc               = validate(model, val_loader, criterion, device)
     

        # Check if this is the best model so far
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            # Save the model state dictionary
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, save_path)
            status = "--> Model Saved!"
        else:
            status = ""

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% {status}")

# Hyperparameters
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Good for small hand datasets
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
train_csv                           = '/path/train_data.csv'
test_csv                            = '/path/test_data.csv'

train_dataset                       = PalmVeinDataset(csv_file=train_csv, transform=data_transforms)
val_dataset                         = PalmVeinDataset(csv_file=test_csv, transform=data_transforms)

train_loader                        = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader                          = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

save_path                           = '/path/best.pt' ### path of saved weight

######## Initialize Model, Loss, Optimizer ############
model                               = DenseNet161_GGA_Binary().to(DEVICE)
# model                               = DesnseNet161().to(DEVICE)
# model                               = ResNet101_GGA_Binary().to(DEVICE)
# model                               = ResNet101.to(DEVICE)

############# Training model ##############
run_training(model, train_loader, val_loader, epochs=200, save_path=save_path)

############# Model evaluation #############
model                               = load_model_weights(model, save_path, DEVICE)
metrics                             = evaluate_gender_metrics(model, val_loader, DEVICE)
metrics                             = evaluate_gender_metrics(model, train_loader, DEVICE)
metrics                             = evaluate_gender_metrics_all(model, train_loader,val_loader, DEVICE)
