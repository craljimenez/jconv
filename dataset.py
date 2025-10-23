import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet, ImageFolder
from torchvision.datasets.folder import default_loader

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


class FolderDatasetWrapper(ImageFolder):
    """Wrapper over torchvision.datasets.ImageFolder to apply the same transform pipeline as FolderDatasetTransforms."""
    def __init__(self, root, transform=None, target_transform=None, loader=None, is_valid_file=None):
        loader = loader or default_loader

        super().__init__(
            root=root,
            transform=None,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.image_transform = transform or transforms.ToTensor()

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.image_transform:
            sample = self.image_transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
