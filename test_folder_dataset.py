import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import FolderDatasetTransforms, FolderDatasetWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a folder dataset and plot 8 sample images with class labels."
    )
    parser.add_argument("root", type=str, help="Root directory of the dataset.")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Image resize dimension applied through FolderDatasetTransforms (default: 256).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of samples to plot (default: 8).",
    )
    return parser.parse_args()


def unnormalize(image_tensor):
    # ImageFolderTransform uses ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype, device=image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype, device=image_tensor.device)
    return image_tensor * std[:, None, None] + mean[:, None, None]


def main():
    args = parse_args()
    root_path = Path(args.root)

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    transform = FolderDatasetTransforms(size=args.size)
    dataset = FolderDatasetWrapper(root=str(root_path), transform=transform)

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in dataset root: {root_path}")

    num_samples = min(args.count, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, steps=num_samples).long().tolist()

    fig, axes = plt.subplots(2, (num_samples + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        image_tensor, target = dataset[idx]
        image = unnormalize(image_tensor).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        class_name = dataset.classes[target]

        ax.imshow(image)
        ax.set_title(f"Idx: {idx} | Target: {target} | Class: {class_name}")
        ax.axis("off")

    # Hide unused axes if count < grid size
    for ax in axes[num_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
