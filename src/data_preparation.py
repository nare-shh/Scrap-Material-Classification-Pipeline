import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class TrashNetDataset(Dataset):

    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
       
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(is_training=True):
   
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
    
    
    ds = load_dataset("garythung/trashnet")
    
    
    class_names = ds['train'].features['label'].names
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(ds['train'])}")
    
    
    train_val_test = ds['train'].train_test_split(test_size=0.3, seed=42)
    val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=42)
    
    train_data = train_val_test['train']
    val_data = val_test['train']
    test_data = val_test['test']
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    train_dataset = TrashNetDataset(
        train_data, 
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = TrashNetDataset(
        val_data, 
        transform=get_transforms(is_training=False)
    )
    
    test_dataset = TrashNetDataset(
        test_data, 
        transform=get_transforms(is_training=False)
    )
    
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
    
    print(f" Train batches: {len(train_loader)}")
    print(f" Val batches: {len(val_loader)}")
    print(f" Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    
    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(batch_size=16)
    
    
    images, labels = next(iter(train_loader))
    print(f"\n Batch shape: {images.shape}")
    print(f" Labels shape: {labels.shape}")
    print(f" Sample labels: {labels[:5]}")