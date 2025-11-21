import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import json
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset import (
    PetDatasetTransforms,
    PetDatasetWrapper,
    PetClassificationTransforms,
    PetClassificationWrapper,
    FolderDatasetTransforms,
    FolderDatasetWrapper,
)
from utils import (
    build_unet,
    build_unet_hybrid_jenc,
    build_fcn,
    build_fcn_hybrid_jenc,
    build_jvgg19,
    build_jvgg21,
    build_vgg19,
    build_vgg21,
    dice_score,
    iou_score,
    get_device,
)

def train_one_epoch_seg(model, loader, optimizer, loss_fn, device):
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


def evaluate_seg(model, loader, loss_fn, n_classes, device):
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


def train_one_epoch_cls(model, loader, optimizer, loss_fn, device):
    """Runs a single training epoch for classification models."""
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loop:
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        loop.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

    avg_loss = total_loss / len(loader)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def evaluate_cls(model, loader, loss_fn, device, score='f1'):
    """
    Evalúa modelos de clasificación usando diferentes métricas de sklearn.

    Args:
        model: El modelo a evaluar.
        loader: DataLoader para el conjunto de validación.
        loss_fn: La función de pérdida.
        device: El dispositivo (CPU/GPU).
        score (str): La métrica a calcular ('accuracy', 'f1', 'precision', 'recall').

    Returns:
        tuple: (pérdida_promedio, valor_de_la_métrica)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating")
        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)

    # Calcular la métrica solicitada usando sklearn
    if score == 'accuracy':
        metric_value = accuracy_score(all_targets, all_preds)
    elif score == 'f1':
        metric_value = f1_score(all_targets, all_preds, average='macro')
    else:
        raise ValueError(f"Métrica de evaluación desconocida: '{score}'. Use 'accuracy' o 'f1'.")

    return avg_loss, metric_value


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

    classification_models = {'jvgg19', 'jvgg21','vgg19','vgg21'}
    is_classification = args.model in classification_models

    # --- 1. Dataset and DataLoaders ---
    if is_classification:
        print("train_dir given:",args.train_dir)
        if args.train_dir:
            if not os.path.isdir(args.train_dir):
                raise FileNotFoundError(f"The provided training directory does not exist or is not a directory: {args.train_dir}")

            transform = FolderDatasetTransforms(size=args.img_size)
            full_dataset = FolderDatasetWrapper(args.train_dir, transform=transform)
            if args.val_dir:
                print("Using provided folder splits for training and validation.")
                train_dataset = full_dataset
                val_dataset = FolderDatasetWrapper(args.val_dir, transform=transform)
            else:
                split_ratio = args.holdout_split
                if split_ratio is None:
                    raise ValueError("Provide --val-dir or set --holdout-split to enable automatic validation split.")
                if not (0.0 < split_ratio < 1.0):
                    raise ValueError("--holdout-split must be between 0 and 1.")
                val_size = max(1, int(len(full_dataset) * split_ratio))
                train_size = len(full_dataset) - val_size
                if train_size == 0:
                    raise ValueError("Holdout split too large; no samples left for training.")
                generator = torch.Generator().manual_seed(args.holdout_seed)
                train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
                print(f"Using holdout split: {train_size} training / {val_size} validation samples (ratio {split_ratio:.2f}).")
        else:
            if args.val_dir:
                raise ValueError("--val-dir provided without --train-dir. Provide training directory or rely on Oxford-IIIT Pet splits.")
            print("Using Oxford-IIIT Pet classification split.")
            cls_transform = PetClassificationTransforms(size=args.img_size)
            train_dataset = PetClassificationWrapper(root=args.data_root, split='trainval', transform=cls_transform, download=True)
            val_dataset = PetClassificationWrapper(root=args.data_root, split='test', transform=cls_transform, download=True)
        
        base_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
        dataset_classes = getattr(base_dataset, 'classes', None)
        if args.num_classes is not None:
            n_classes = args.num_classes
        elif dataset_classes is not None:
            n_classes = len(dataset_classes)
        else:
            raise ValueError("Unable to infer number of classes. Please specify --num-classes.")
        if dataset_classes is not None and args.num_classes is not None and n_classes != len(dataset_classes):
            print(f"Warning: --num-classes ({args.num_classes}) does not match dataset classes ({len(dataset_classes)}). Proceeding with --num-classes.")
        class_names = dataset_classes if dataset_classes is not None else [str(i) for i in range(n_classes)]
        print(f"Number of classes: {n_classes}")
    else:
        dataset_transforms = PetDatasetTransforms(size=args.img_size)
        train_dataset = PetDatasetWrapper(root=args.data_root, split='trainval', transform=dataset_transforms, download=True)
        val_dataset = PetDatasetWrapper(root=args.data_root, split='test', transform=dataset_transforms, download=True)
        n_classes = 3  # 0: Background, 1: Pet, 2: Border
        class_names = ['Background', 'Pet', 'Border']

    
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")

    # --- 2. Model ---
    if args.model == 'unet':
        print("Building standard UNet model...")
        model = build_unet(in_ch=3, n_classes=n_classes, base_ch=args.base_pos, depth=args.depth, bilinear=True)
    elif args.model == 'unet_hybrid':
        print(f"Building UNet-Hybrid model with base_pos={args.base_pos} and base_neg={args.base_neg}...")
        print(f"\tUsing activation='{args.activation}' and orth={args.orth}")
        model = build_unet_hybrid_jenc(
            in_ch=3,
            n_classes=n_classes,
            base_pos=args.base_pos,
            base_neg=args.base_neg,
            depth=args.depth,
            act=args.activation,
            orth=args.orth,
        )
    elif args.model == 'fcn':
        print("Building standard FCN model...")
        model = build_fcn(in_ch=3, n_classes=n_classes, base_ch=args.base_pos, stages=args.depth)
    elif args.model == 'fcn_hybrid':
        print(f"Building FCN-Hybrid model with base_pos={args.base_pos} and base_neg={args.base_neg}...")
        model = build_fcn_hybrid_jenc(
            in_ch=3,
            n_classes=n_classes,
            base_pos=args.base_pos,
            base_neg=args.base_neg,
            stages=args.depth,
            act=args.activation,
        )
    elif args.model == 'jvgg19':
        print(f"Building JVGG19 classifier with base_pos={args.base_pos}, base_neg={args.base_neg}, orth={args.orth}")
        model = build_jvgg19(
            in_ch=3,
            base_pos=args.base_pos,
            base_neg=args.base_neg,
            n_classes=n_classes,
            act=args.activation,
            orth=args.orth,
            proj_mode=args.proj_mode,
            avgpool_size=args.avgpool_size,
            dropout=args.dropout,
        )
    elif args.model == 'jvgg21':
        print(f"Building JVGG21 classifier with base_pos={args.base_pos}, base_neg={args.base_neg}, orth={args.orth}")
        model = build_jvgg21(
            in_ch=3,
            base_pos=args.base_pos,
            base_neg=args.base_neg,
            n_classes=n_classes,
            act=args.activation,
            orth=args.orth,
            proj_mode=args.proj_mode,
            avgpool_size=args.avgpool_size,
            dropout=args.dropout,
        )
    elif args.model == 'vgg19':
        print(f"Building VGG19 classifier with base_ch={args.base_pos}")
        model = build_vgg19(
            in_ch=3,
            base_ch=args.base_pos,
            n_classes=n_classes,
            avgpool_size=args.avgpool_size,
            dropout=args.dropout,
        )
    elif args.model == 'vgg21':
        print(f"Building VGG21 classifier with base_ch={args.base_pos}")
        model = build_vgg21(
            in_ch=3,
            base_pos=args.base_pos,
            n_classes=n_classes,
            avgpool_size=args.avgpool_size,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: '{args.model}'")

    model.to(device)

    # --- 3. Loss and Optimizer ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # --- 4. Training Loop ---
    best_val_metric = -1.0
    training_history = []
    log_path = f"training_log_{args.model}.json"
    metric_label = args.metric_evaluating.title() if is_classification else "mIoU"

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        if is_classification:
            train_loss, train_acc = train_one_epoch_cls(model, train_loader, optimizer, loss_fn, device)
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")

            val_loss, val_acc = evaluate_cls(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)
            print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f} | Val {metric_label}: {val_acc * 100:.2f}%")

            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                f'val_{args.metric_evaluating}': val_acc,
            }
            current_metric = val_acc
        else:
            train_loss = train_one_epoch_seg(model, train_loader, optimizer, loss_fn, device)
            print(f"Epoch {epoch + 1} - Average Training Loss: {train_loss:.4f}")

            val_loss, val_dice, val_iou = evaluate_seg(model, val_loader, loss_fn, n_classes, device)
            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1} - Average Validation Loss: {val_loss:.4f}")
            print("--- Validation Metrics ---")
            for i in range(n_classes):
                print(f"  Class '{class_names[i]}':")
                print(f"    - Dice: {val_dice[i]:.4f}")
                print(f"    - IoU:  {val_iou[i]:.4f}")

            mean_iou = val_iou.mean().item()
            mean_dice = val_dice.mean().item()
            print(f"\n  Mean IoU (mIoU): {mean_iou:.4f}")
            print(f"  Mean Dice:     {mean_dice:.4f}")
            print("--------------------------")

            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mean_iou': mean_iou,
                'mean_dice': mean_dice,
                'per_class_metrics': {
                    class_names[i]: {
                        'iou': val_iou[i].item(),
                        'dice': val_dice[i].item(),
                    }
                    for i in range(n_classes)
                },
            }
            current_metric = mean_iou

        training_history.append(epoch_log)

        with open(log_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        print(f"\nTraining history saved to {log_path}")

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"best_model_{args.model}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Validation {metric_label} improved. Saved model to {model_path}")

    # --- 5. Save validation loader configuration ---
    if args.save_val_loader_config:
        val_loader_config = {
            "model": args.model,
            "data_root": args.data_root,
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_classes": n_classes,
            "best_model_path": os.path.join("models", f"best_model_{args.model}.pth")
        }
        config_path = os.path.join("models", f"val_loader_config_{args.model}.json")
        with open(config_path, 'w') as f:
            json.dump(val_loader_config, f, indent=4)
        print(f"\nValidation loader configuration saved to {config_path}")
        print("You can now use test.py with this configuration to evaluate the model.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train segmentation and classification models on Oxford-IIIT Pet or custom datasets.")
    parser.add_argument(
        '--model',
        type=str,
        default='unet_hybrid',
        choices=['unet', 'unet_hybrid', 'fcn', 'fcn_hybrid', 'jvgg19', 'jvgg21','vgg19', 'vgg21'],
        help="Model to train.",
    )
    parser.add_argument(
        '--from-best-params',
        type=str,
        default=None,
        help="Load hyperparameters from the JSON file for the specified model",
    )
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for training and evaluation.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--img-size', type=int, default=256, help="Size to resize input images to.")
    parser.add_argument(
        '--gpu-mem-fraction',
        type=float,
        default=None,
        help="Fraction of GPU memory to use (e.g., 0.8 for 80%). Only for CUDA devices.",
    )
    parser.add_argument('--depth', type=int, default=4, help="Model depth (or stages for FCN/UNet).")
    parser.add_argument(
        '--base-pos',
        type=int,
        default=32,
        help="Positive channels for J-Conv encoder/backbone (used by hybrid and JVGG models).",
    )
    parser.add_argument(
        '--base-neg',
        type=int,
        default=8,
        help="Negative channels for J-Conv encoder/backbone (used by hybrid and JVGG models).",
    )
    parser.add_argument(
        '--activation',
        type=str,
        default="tanh",
        choices=['tanh', 'gelu', 'leaky_relu'],
        help="Activation function to use in JConv blocks.",
    )
    parser.add_argument(
        '--orth',
        action='store_true',
        help="Use orthogonal J-Conv layers (JConv2dOrtho) instead of standard J-Conv layers.",
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help="Root path for Oxford-IIIT Pet datasets (segmentation/classification).",
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default=None,
        help="Path to folder dataset (training split) for classification. Provide with --val-dir.",
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default=None,
        help="Path to folder dataset (validation split) for classification. Provide with --train-dir.",
    )
    parser.add_argument(
        '--holdout-split',
        type=float,
        default=0.2,
        help="Holdout ratio for automatic validation split when only --train-dir is provided.",
    )
    parser.add_argument(
        '--holdout-seed',
        type=int,
        default=42,
        help="Random seed used for holdout splitting.",
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help="Number of classes for classification. Inferred from dataset when not provided.",
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        '--proj-mode',
        type=str,
        default='sub',
        choices=['sub', 'concat'],
        help="Projection mode from Pontryagin space to Euclidean (JVGG models).",
    )
    parser.add_argument(
        '--avgpool-size',
        type=int,
        default=7,
        help="Output size of the adaptive average pooling layer in JVGG classifiers.",
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help="Dropout probability within the JVGG classifier head.",
    )
    parser.add_argument(
        "--metric-evaluating",
        type=str,
        default="accuracy",
        help="Metric to evaluate validation data in classification task."
        
    )
    parser.add_argument(
        '--save-val-loader-config',
        action='store_true',
        help="Save the configuration required to recreate the validation loader for future testing."
    )
    args = parser.parse_args()
    main(args)
