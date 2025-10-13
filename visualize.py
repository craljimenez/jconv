import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from matplotlib.colors import Normalize
from utils import build_unet, build_unet_hybrid_jenc, dice_score, iou_score
from dataset import PetDatasetTransforms, PetDatasetWrapper

def denorm(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)

def visualize_results(unet, unet_hybrid, val_loader, device, num_images=5):
    unet.eval()
    unet_hybrid.eval()

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= num_images:
                break

            x = x.to(device)
            y = y.to(device)

            # Predictions
            out_unet = unet(x)
            pred_unet = torch.argmax(out_unet, dim=1)
            

            out_unet_hybrid = unet_hybrid(x)
            pred_unet_hybrid = torch.argmax(out_unet_hybrid, dim=1)
            # Print unique values in predictions for debugging
            print(f"Unique values in pred_unet: {torch.unique(pred_unet)}")
            print(f"Unique values in pred_unet_hybrid: {torch.unique(pred_unet_hybrid)}")
            # Calculate metrics
            n_classes = 3
            dice_unet = dice_score(out_unet, y, n_classes, device)
            iou_unet = iou_score(out_unet, y, n_classes, device)
            dice_hybrid = dice_score(out_unet_hybrid, y, n_classes, device)
            iou_hybrid = iou_score(out_unet_hybrid, y, n_classes, device)

            # Denormalize original image for display
            x_denorm = denorm(x.cpu(), torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))

            # Plotting
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))

            # Original Image
            plt.subplot(1, 4, 1)
            plt.imshow(x_denorm[0].permute(1, 2, 0))
            plt.title("Original Image")
            plt.axis("off")

            # Define normalization for mask display
            norm = Normalize(vmin=0, vmax=2)

            # Original Mask
            plt.subplot(1, 4, 2)
            plt.imshow(y.cpu()[0], cmap='nipy_spectral', norm=norm)
            plt.title("Original Mask")
            plt.axis("off")

            # Predicted Mask - UNet
            plt.subplot(1, 4, 3)
            plt.imshow(pred_unet.cpu()[0], cmap='nipy_spectral', norm=norm)
            title_unet = (f"UNet Prediction\n"
                          f"mIoU: {iou_unet.mean():.3f} | mDice: {dice_unet.mean():.3f}")
            plt.title(title_unet)
            plt.axis("off")

            # Predicted Mask - UNet Hybrid
            plt.subplot(1, 4, 4)
            plt.imshow(pred_unet_hybrid.cpu()[0], cmap='nipy_spectral', norm=norm)
            title_hybrid = (f"UNet Hybrid Prediction\n"
                            f"mIoU: {iou_hybrid.mean():.3f} | mDice: {dice_hybrid.mean():.3f}")
            plt.title(title_hybrid)
            plt.axis("off")

            plt.show()

if __name__ == "__main__":
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1 # Process one image at a time for visualization
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    PIN_MEMORY = True
    DATA_DIR = "./data"
    N_CLASSES = 3

    # Load Data
    dataset_transforms = PetDatasetTransforms(size=IMAGE_HEIGHT)
    val_dataset = PetDatasetWrapper(root=DATA_DIR, split='test', transform=dataset_transforms, download=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=PIN_MEMORY)


    # Load Models
    unet_model = build_unet(in_ch=3, n_classes=N_CLASSES, bilinear=True, base_ch=32).to(DEVICE)
    unet_hybrid_model = build_unet_hybrid_jenc(in_ch=3, n_classes=N_CLASSES).to(DEVICE)

    # Load trained weights
    # Make sure you have the trained model weights in the root directory
    WEIGHTS_PATH = "./models/"
    try:
        unet_model.load_state_dict(torch.load(f"{WEIGHTS_PATH}best_model_unet.pth", map_location=DEVICE))
        unet_hybrid_model.load_state_dict(torch.load(f"{WEIGHTS_PATH}best_model_unet_hybrid.pth", map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure you have the trained model weights 'best_model_unet.pth' and 'best_model_unet_hybrid.pth' in the '{WEIGHTS_PATH}' directory.")
        exit()


    # Visualize
    visualize_results(unet_model, unet_hybrid_model, val_loader, DEVICE)