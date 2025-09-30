"""
Data Preparation Module
Loads TrashNet dataset, applies augmentation, and creates train/val/test splits
"""

import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class TrashNetDataset(Dataset):
    """Custom Dataset for TrashNet"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(is_training=True):
    """
    Get data transforms for training and validation
    
    Args:
        is_training: If True, applies augmentation
    
    Returns:
        torchvision transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def prepare_dataloaders(batch_size=32, num_workers=2):
    """
    Load dataset and create dataloaders
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    print("📦 Loading TrashNet dataset from HuggingFace...")
    
    # Load dataset
    ds = load_dataset("garythung/trashnet")
    
    # Get class names
    class_names = ds['train'].features['label'].names
    print(f"✅ Classes: {class_names}")
    print(f"📊 Train samples: {len(ds['train'])}")
    print(f"📊 Test samples: {len(ds['test'])}")
    
    # Split train into train and validation (90-10 split)
    train_val_split = ds['train'].train_test_split(test_size=0.1, seed=42)
    
    # Create datasets
    train_dataset = TrashNetDataset(
        train_val_split['train'], 
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = TrashNetDataset(
        train_val_split['test'], 
        transform=get_transforms(is_training=False)
    )
    
    test_dataset = TrashNetDataset(
        ds['test'], 
        transform=get_transforms(is_training=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(batch_size=16)
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Batch shape: {images.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print(f"✅ Sample labels: {labels[:5]}")