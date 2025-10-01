import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

from data_preparation import prepare_dataloaders


class MaterialClassifier(nn.Module):
    
    def __init__(self, num_classes=6):
        super(MaterialClassifier, self).__init__()
        
        self.model = models.mobilenet_v2(pretrained=True)
        
        for param in self.model.features[:-3].parameters():
            param.requires_grad = False
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_labels, all_predictions


def plot_training_history(history, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_history.png', dpi=300, bbox_inches='tight')
    print(f" Saved training history to {save_path}/training_history.png")


def plot_confusion_matrix(labels, predictions, class_names, save_path='results'):
    
    os.makedirs(save_path, exist_ok=True)
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f" Saved confusion matrix to {save_path}/confusion_matrix.png")


def train_model(num_epochs=15, batch_size=32, learning_rate=0.001):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(batch_size=batch_size)
    
    model = MaterialClassifier(num_classes=len(class_names)).to(device)
    print(f" Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f" Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"\n Starting training for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_acc, val_labels, val_predictions = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, 'models/best_model.pth')
            print(f" Saved best model with Val Acc: {val_acc:.2f}%\n")
    
    plot_training_history(history)
    
    test_loss, test_acc, test_labels, test_predictions = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    plot_confusion_matrix(test_labels, test_predictions, class_names)
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, target_names=class_names))
    
    torch.save(model.state_dict(), 'models/final_model.pth')
    print("Training complete")
    
    return model, class_names


if __name__ == "__main__":
    model, class_names = train_model(num_epochs=15, batch_size=32, learning_rate=0.001)