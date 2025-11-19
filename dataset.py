import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

import os
from PIL import Image
from torch.utils.data import Dataset

class PetDatasetTransforms:
    """Transforms for the Oxford-IIIT Pet Dataset."""
    def __init__(self, size=256):
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __call__(self, img, mask):
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        
        # Convert mask to tensor and remap class values
        # Original: 1: Pet, 2: Border, 3: Background
        # New:      0: Background, 1: Pet, 2: Border
        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask == 3] = 0 # Background
        # Pet (1) and Border (2) keep their values
        
        return img, mask


class PetDatasetWrapper(OxfordIIITPet):
    """A wrapper for the OxfordIIITPet dataset to apply transforms to both image and mask."""
    def __init__(self, root, split, transform=None, download=False):
        super().__init__(root=root, split=split, target_types='segmentation', download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


class PetClassificationTransforms:
    """Transforms for Oxford-IIIT Pet classification."""
    def __init__(self, size=256):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img)


class PetClassificationWrapper(OxfordIIITPet):
    """Wrapper for Oxford-IIIT Pet classification setting."""
    def __init__(self, root, split, transform=None, download=False):
        super().__init__(root=root, split=split, target_types='category', download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transform:
            img = self.transform(img)
        target = torch.as_tensor(target, dtype=torch.long)
        return img, target


class FolderDatasetTransforms:
    """
    Image transformations for folder-based datasets where each sub-folder corresponds to a class.

    Examples
    --------
    dataset_root/
        Healthy/
            img1.png
            ...
        Mosaic/
            ...
    """
    def __init__(self, size=256):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img)

class FolderDatasetWrapper(Dataset):
    """
    A custom PyTorch Dataset for image classification datasets 
    where sub-folders represent class labels.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initializes the Dataset. This function scans the directory 
        and builds a list of (image_path, label_index) tuples.
        
        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied 
                                            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # --- 1. Scan and Map Classes (Done ONCE at initialization) ---
        print(f"Scanning dataset in: {root_dir}")
        classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        
        for i, class_name in enumerate(classes):
            self.class_to_idx[class_name] = i
            class_path = os.path.join(root_dir, class_name)
            
            # Find all image files in the sub-folder
            sub_folders = os.listdir(class_path)
            for filename in sub_folders:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(class_path, filename)
                    self.image_paths.append(path)
                    self.labels.append(i) # Store the integer index
        self.classes = classes
        print(f"Found {len(self.image_paths)} images across {len(classes)} classes.")


        


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)


    def __getitem__(self, idx):
        """
        Loads and returns one sample (image and label) from the dataset.
        This is the method responsible for LAZY LOADING.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            tuple: (image, label) where image is a Tensor and label is an integer.
        """
        # --- 2. Load Image Data (Done ON-DEMAND) ---
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # --- 3. Apply Transformations ---
        if self.transform:
            image = self.transform(image)

        return image, label