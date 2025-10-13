import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

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