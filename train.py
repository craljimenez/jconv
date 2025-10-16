import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os

from dataset import PetDatasetTransforms, PetDatasetWrapper
from utils import build_unet, build_unet_hybrid_jenc, build_fcn, build_fcn_hybrid_jenc, dice_score, iou_score


def get_device():
    """Gets the best available device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Runs a single training epoch."""
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0

    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluate(model, loader, loss_fn, n_classes, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    total_dice = torch.zeros(n_classes, device=device)
    total_iou = torch.zeros(n_classes, device=device)
    
    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            # Calculate metrics
            total_dice += dice_score(outputs, masks, n_classes, device)
            total_iou += iou_score(outputs, masks, n_classes, device)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)

    return avg_loss, avg_dice, avg_iou


def main(args):
    """Main function to run the training and evaluation."""
    device = get_device()
    print(f"Using device: {device}")

    # --- Load hyperparameters from JSON if specified ---
    if args.from_best_params:
        json_path = f'{args.from_best_params}'
        print(f"Loading best hyperparameters from '{json_path}'...")
        try:
            with open(json_path, 'r') as f:
                best_params_data = json.load(f)
            
            # Override args with the loaded hyperparameters
            loaded_params = best_params_data['best_hyperparameters']
            for key, value in loaded_params.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            
            # The model name should match the one from the file
            

            print("Successfully loaded and applied the following hyperparameters:")
            for key, value in loaded_params.items():
                print(f"  - {key}: {value}")

        except FileNotFoundError:
            print(f"Error: Hyperparameter file '{json_path}' not found. Please run Bayesian optimization first.")
            return

    # --- Set GPU Memory Limit ---
    if device.type == 'cuda' and args.gpu_mem_fraction is not None:
        if 0.0 < args.gpu_mem_fraction <= 1.0:
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device)
            print(f"Limiting GPU memory on {device} to {args.gpu_mem_fraction * 100:.0f}% of total capacity.")
        else:
            print("Warning: --gpu-mem-fraction must be between 0.0 and 1.0. Ignoring.")

    # --- 1. Dataset and DataLoaders ---
    dataset_transforms = PetDatasetTransforms(size=args.img_size)
    train_dataset = PetDatasetWrapper(root='./data', split='trainval', transform=dataset_transforms, download=True)
    val_dataset = PetDatasetWrapper(root='./data', split='test', transform=dataset_transforms, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")

    # --- 2. Model ---
    n_classes = 3 # 0: Background, 1: Pet, 2: Border
    if args.model == 'unet':
        print("Building standard UNet model...")
        model = build_unet(in_ch=3, n_classes=n_classes, base_ch=args.base_pos, depth=args.depth, bilinear=True)
    elif args.model == 'unet_hybrid':
        print(f"Building UNet-Hybrid model with base_pos={args.base_pos} and base_neg={args.base_neg}...")
        print(f"\t Using activation='{args.activation}' and orth={args.orth}")
        model = build_unet_hybrid_jenc(in_ch=3,
                                       n_classes=n_classes,
                                       base_pos=args.base_pos,
                                       base_neg=args.base_neg,
                                       depth=args.depth,
                                       act=args.activation,
                                       orth=args.orth
                                       )
    elif args.model == 'fcn':
        print("Building standard FCN model...")
        model = build_fcn(in_ch=3, n_classes=n_classes, base_ch=args.base_pos, stages=args.depth)
    elif args.model == 'fcn_hybrid':
        print(f"Building FCN-Hybrid model with base_pos={args.base_pos} and base_neg={args.base_neg}...")
        model = build_fcn_hybrid_jenc(in_ch=3,
                                      n_classes=n_classes,
                                      base_pos=args.base_pos,
                                      base_neg=args.base_neg,
                                      stages=args.depth,
                                      act=args.activation
                                      )
    else:
        raise ValueError(f"Unknown model type: '{args.model}'")

    model.to(device)

    # --- 3. Loss and Optimizer ---
    # Note: For multi-class, CrossEntropyLoss is standard.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # --- 4. Training Loop ---
    best_val_iou = -1.0
    class_names = ['Background', 'Pet', 'Border']
    training_history = []
    # To save fitting logs.
    log_path = f"training_log_{args.model}.json" 
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1} - Average Training Loss: {train_loss:.4f}")

        val_loss, val_dice, val_iou = evaluate(model, val_loader, loss_fn, n_classes, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} - Average Validation Loss: {val_loss:.4f}")
        print("--- Validation Metrics ---")
        for i in range(n_classes):
            print(f"  Class '{class_names[i]}':")
            print(f"    - Dice: {val_dice[i]:.4f}")
            print(f"    - IoU:  {val_iou[i]:.4f}")
        
        mean_iou = val_iou.mean()
        print(f"\n  Mean IoU (mIoU): {mean_iou:.4f}")
        print(f"  Mean Dice:     {val_dice.mean():.4f}")
        print("--------------------------")

        # Log results for this epoch
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'mean_iou': mean_iou.item(),
            'mean_dice': val_dice.mean().item(),
            'per_class_metrics': {
                class_names[i]: {
                    'iou': val_iou[i].item(),
                    'dice': val_dice[i].item()
                } for i in range(n_classes)
            }
        }
        training_history.append(epoch_log)

        # --- 5. Save Training History ---
        with open(log_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        print(f"\nTraining history saved to {log_path}")
        
        # Save the best model based on mean IoU
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"best_model_{args.model}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Validation mIoU improved. Saved model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train U-Net models on Oxford-IIIT Pet Dataset.")
    parser.add_argument('--model', type=str, default='unet_hybrid', choices=['unet', 'unet_hybrid', 'fcn', 'fcn_hybrid'],
                        help="Model to train ('unet', 'unet_hybrid', 'fcn', 'fcn_hybrid').")
    parser.add_argument('--from-best-params', type=str, default=None,
                        help="Load hyperparameters from the JSON file for the specified model")
    parser.add_argument('--epochs', type=int, default=25,
                        help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=8,
                        help="Batch size for training and evaluation.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--img-size', type=int, default=256,
                        help="Size to resize input images to.")
    parser.add_argument('--gpu-mem-fraction', type=float, default=None,
                        help="Fraction of GPU memory to use (e.g., 0.8 for 80%). Only for CUDA devices.")
    parser.add_argument('--depth', type=int, default=4,
                        help="Model depth (or stages for FCN).")
    parser.add_argument('--base-pos', type=int, default=32,
                        help="Positive channels for J-Conv encoder (only for unet_hybrid).")
    parser.add_argument('--base-neg', type=int, default=8,
                        help="Negative channels for J-Conv encoder (only for unet_hybrid).")
    parser.add_argument('--activation',type=str,default="tanh",choices=['tanh','gelu','leaky_relu'],
                        help="Activation function to use in JCONV Blocks")
    parser.add_argument('--orth', action='store_true',
                        help="Use orthogonal J-Conv layers (JConv2dOrtho) instead of standard J-Conv layers.")
    args = parser.parse_args()
    main(args)