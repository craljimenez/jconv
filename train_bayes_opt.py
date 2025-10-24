import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
import csv
from skopt import gp_minimize
from skopt.plots import plot_objective, plot_evaluations
from skopt.space import Real, Integer, Categorical
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
    dice_score,
    iou_score,
)


def get_device():
    """Gets the best available device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


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
    """Runs a single classification training epoch."""
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


def evaluate_cls(model, loader, loss_fn, device):
    """Evaluates classification models."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="Evaluating")
        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(loader)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc

# Global variable to keep track of trials
trial_num = 0

# Global variables for optimization
search_space_dims = []
fixed_params = {}
csv_log_filename = ""

def objective(params):
    """Objective function for Bayesian Optimization."""
    global trial_num, csv_log_filename
    trial_num += 1

    opt_params = {dim.name: val for dim, val in zip(search_space_dims, params)}
    all_params = {**fixed_params, **opt_params}

    if 'base_pos_factor' in all_params:
        all_params['base_pos'] = 2 ** all_params.pop('base_pos_factor')
    if 'base_neg_factor' in all_params:
        all_params['base_neg'] = 2 ** all_params.pop('base_neg_factor')

    classification_models = {'jvgg19', 'jvgg21'}
    is_classification = args.model in classification_models

    lr = float(all_params['lr'])
    batch_size = int(all_params['batch_size'])
    if 'base_pos' in all_params:
        all_params['base_pos'] = int(all_params['base_pos'])
    if 'base_neg' in all_params:
        all_params['base_neg'] = int(all_params['base_neg'])
    if 'depth' in all_params:
        all_params['depth'] = int(all_params['depth'])
    if 'dropout' in all_params:
        all_params['dropout'] = float(all_params['dropout'])
    activation = all_params.get('activation', args.activation if args.activation is not None else 'tanh')
    orth = args.orth
    num_epochs_per_trial = args.num_epochs_per_trial
    device = get_device()
    pin_memory = device.type == 'cuda'
    metric_label = "Accuracy" if is_classification else "mIoU"

    print(f"\n--- Bayesian Opt Trial: {trial_num} ---")

    base_neg = None
    dropout = None
    best_val_metric = -1.0

    try:
        if is_classification:
            base_pos = all_params['base_pos']
            base_neg = all_params.get('base_neg', args.base_neg)
            dropout = all_params.get('dropout', args.dropout if args.dropout is not None else 0.5)

            if orth:
                base_neg = base_pos
            if base_neg is not None and base_neg > base_pos:
                print(f"Constraint not met: base_neg ({base_neg}) > base_pos ({base_pos}). Penalizing.")
                return 1.0

            print(
                f"Params: lr={lr:.6f}, base_pos={base_pos}, base_neg={base_neg}, "
                f"batch_size={batch_size}, dropout={dropout:.2f}, activation={activation}, orth={orth}"
            )

            if args.train_dir:
                transform = FolderDatasetTransforms(size=args.img_size)
                full_dataset = FolderDatasetWrapper(args.train_dir, transform=transform)
                if args.val_dir:
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
                cls_transform = PetClassificationTransforms(size=args.img_size)
                train_dataset = PetClassificationWrapper(
                    root=args.data_root, split='trainval', transform=cls_transform, download=True
                )
                val_dataset = PetClassificationWrapper(
                    root=args.data_root, split='test', transform=cls_transform, download=True
                )

            base_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
            dataset_classes = getattr(base_dataset, 'classes', None)
            if args.num_classes is not None:
                n_classes = args.num_classes
            elif dataset_classes is not None:
                n_classes = len(dataset_classes)
            else:
                raise ValueError("Unable to infer number of classes. Provide --num-classes.")

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )

            if args.model == 'jvgg19':
                model = build_jvgg19(
                    in_ch=3,
                    base_pos=base_pos,
                    base_neg=base_neg,
                    n_classes=n_classes,
                    act=activation,
                    orth=orth,
                    proj_mode=args.proj_mode,
                    avgpool_size=args.avgpool_size,
                    dropout=dropout,
                )
            elif args.model == 'jvgg21':
                model = build_jvgg21(
                    in_ch=3,
                    base_pos=base_pos,
                    base_neg=base_neg,
                    n_classes=n_classes,
                    act=activation,
                    orth=orth,
                    proj_mode=args.proj_mode,
                    avgpool_size=args.avgpool_size,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown classification model type: '{args.model}'")

            model.to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_val_metric = -1.0
            for epoch in range(num_epochs_per_trial):
                print(f"\n--- Epoch {epoch + 1}/{num_epochs_per_trial} ---")
                train_one_epoch_cls(model, train_loader, optimizer, loss_fn, device)
                _, val_acc = evaluate_cls(model, val_loader, loss_fn, device)
                print(f"Epoch {epoch + 1} - Val Acc: {val_acc:.4f}")
                if val_acc > best_val_metric:
                    best_val_metric = val_acc

        else:
            depth = all_params['depth']
            base_pos = all_params['base_pos']
            base_neg = None
            if args.model in ('unet_hybrid', 'fcn_hybrid'):
                base_neg = all_params.get('base_neg', None)
                if orth:
                    base_neg = base_pos
                if base_neg is not None and base_neg > base_pos:
                    print(f"Constraint not met: base_neg ({base_neg}) > base_pos ({base_pos}). Penalizing.")
                    return 1.0

            mode = all_params.get('mode', None)
            print(
                f"Params: lr={lr:.6f}, depth={depth}, base_pos={base_pos}, base_neg={base_neg}, "
                f"batch_size={batch_size}, activation={activation}, orth={orth}, mode={mode}"
            )

            dataset_transforms = PetDatasetTransforms(size=args.img_size)
            train_dataset = PetDatasetWrapper(
                root=args.data_root, split='trainval', transform=dataset_transforms, download=True
            )
            val_dataset = PetDatasetWrapper(
                root=args.data_root, split='test', transform=dataset_transforms, download=True
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )

            n_classes = 3
            if args.model == 'unet':
                model = build_unet(in_ch=3, n_classes=n_classes, base_ch=base_pos, depth=depth, bilinear=True)
            elif args.model == 'unet_hybrid':
                model = build_unet_hybrid_jenc(
                    in_ch=3,
                    n_classes=n_classes,
                    base_pos=base_pos,
                    base_neg=base_neg if base_neg is not None else base_pos,
                    depth=depth,
                    act=activation,
                    orth=orth,
                    mode=mode,
                )
            elif args.model == 'fcn':
                model = build_fcn(in_ch=3, n_classes=n_classes, base_ch=base_pos, stages=depth)
            elif args.model == 'fcn_hybrid':
                model = build_fcn_hybrid_jenc(
                    in_ch=3,
                    n_classes=n_classes,
                    base_pos=base_pos,
                    base_neg=base_neg if base_neg is not None else base_pos,
                    stages=depth,
                    act=activation,
                )
            else:
                raise ValueError(f"Unknown model type: '{args.model}'")

            model.to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_val_metric = -1.0
            for epoch in range(num_epochs_per_trial):
                print(f"\n--- Epoch {epoch + 1}/{num_epochs_per_trial} ---")
                train_one_epoch_seg(model, train_loader, optimizer, loss_fn, device)
                _, _, val_iou = evaluate_seg(model, val_loader, loss_fn, n_classes, device)
                mean_iou = val_iou.mean().item()
                print(f"Epoch {epoch + 1} - Val mIoU: {mean_iou:.4f}")
                if mean_iou > best_val_metric:
                    best_val_metric = mean_iou

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"WARNING: CUDA out of memory on trial {trial_num}. Skipping.")
            torch.cuda.empty_cache()
            best_val_metric = -1.0
        else:
            raise e

    log_file = csv_log_filename
    is_new_file = not os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            header = ['trial', 'lr', 'batch_size', 'base_pos', 'base_neg']
            if is_classification:
                header.extend(['dropout', 'best_accuracy'])
            else:
                header.extend(['depth', 'best_miou'])
            writer.writerow(header)

        base_neg_log = base_neg if 'base_neg' in locals() and base_neg is not None else 'NA'
        if is_classification:
            row = [trial_num, lr, batch_size, base_pos, base_neg_log, dropout, best_val_metric]
        else:
            row = [trial_num, lr, batch_size, base_pos, base_neg_log, depth, best_val_metric]
        writer.writerow(row)

    return -best_val_metric

def main(args):
    """Main function to run the Bayesian Optimization."""
    global search_space_dims, fixed_params, csv_log_filename

    # --- Define filenames based on model ---
    csv_log_filename = f'bayes_opt_log_{args.model}.csv'
    json_log_filename = f'best_hyperparameters_{args.model}.json'

    classification_models = {'jvgg19', 'jvgg21'}

    search_space_dims.clear()
    fixed_params.clear()

    # --- Define full search space and identify fixed vs. optimized params ---
    if args.model in ("unet_hybrid", "fcn_hybrid") and not args.orth:
        full_search_space = {
            'lr': Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
            'depth': Integer(3, 5, name='depth'),
            'base_pos': Integer(2, 5, name='base_pos_factor'),
            'base_neg': Integer(2, 4, name='base_neg_factor'),
            'batch_size': Integer(4, 16, name='batch_size'),
            'activation': Categorical(['tanh', 'gelu', 'leaky_relu'], name='activation'),
            'mode': Categorical(['out', 'in', 'output'], name='mode'),
        }
    elif args.model in ("unet_hybrid", "fcn_hybrid"):
        full_search_space = {
            'lr': Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
            'depth': Integer(3, 5, name='depth'),
            'base_pos': Integer(2, 5, name='base_pos_factor'),
            'batch_size': Integer(4, 16, name='batch_size'),
            'activation': Categorical(['tanh', 'gelu', 'leaky_relu'], name='activation'),
        }
    elif args.model in ("jvgg19", "jvgg21") and not args.orth:
        full_search_space = {
            'lr': Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
            'base_pos': Integer(2, 5, name='base_pos_factor'),
            'base_neg': Integer(2, 4, name='base_neg_factor'),
            'batch_size': Integer(8, 32, name='batch_size'),
            'activation': Categorical(['tanh', 'gelu', 'leaky_relu'], name='activation'),
            'dropout': Real(0.1, 0.7, name='dropout'),
        }
    elif args.model in ("jvgg19", "jvgg21"):
        full_search_space = {
            'lr': Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
            'base_pos': Integer(2, 5, name='base_pos_factor'),
            'batch_size': Integer(8, 32, name='batch_size'),
            'activation': Categorical(['tanh', 'gelu', 'leaky_relu'], name='activation'),
            'dropout': Real(0.1, 0.7, name='dropout'),
        }
    else:
        full_search_space = {
            'lr': Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
            'depth': Integer(3, 5, name='depth'),
            'base_pos': Integer(2, 5, name='base_pos_factor'),
            'batch_size': Integer(4, 16, name='batch_size'),
        }

    for name, space in full_search_space.items():
        arg_val = getattr(args, name)
        if arg_val is not None:
            # Special handling for base_pos if it's fixed
            if name in ('base_pos', 'base_neg'):
                if arg_val % 2 != 0:
                    print(f"Warning: Provided --{name.replace('_', '-')} ({arg_val}) is not a multiple of 2. This is unusual but will be used as a fixed value.")
                fixed_params[name] = arg_val
            else:
                fixed_params[name] = arg_val
        else:
            search_space_dims.append(space)

    if not search_space_dims:
        print("All hyperparameters are fixed. No optimization to perform.")
        print("Fixed parameters:", fixed_params)
        # You could run a single training here if desired, but for now we exit.
        return

    print("Starting Bayesian Optimization for hyperparameter tuning...")
    optimizing_names = []
    for dim in search_space_dims:
        optimizing_names.append('base_pos' if dim.name == 'base_pos_factor'\
                                else "base_neg" if dim.name == "base_neg_factor"\
                                else dim.name
                                )

    print("Optimizing for:", optimizing_names)
    print("Fixed parameters:", fixed_params)
    
    # Clean up previous log file if it exists
    if os.path.exists(csv_log_filename):
        os.remove(csv_log_filename)
        print(f"Removed previous log file: {csv_log_filename}")

    # Run optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space_dims,
        n_calls=args.n_calls,
        n_initial_points=5,
        random_state=42,
        acq_func="EI" # Expected Improvement
    )

    # Combine best found params with fixed ones
    best_opt_params = {}
    for dim, val in zip(search_space_dims, result.x):
        if dim.name == 'base_pos_factor':
            best_opt_params['base_pos'] = 2**val
        elif dim.name == 'base_neg_factor':
            best_opt_params['base_neg'] = 2**val
        else:
            best_opt_params[dim.name] = val
    best_params = {**fixed_params, **best_opt_params}

    best_metric = -result.fun
    metric_label = "accuracy" if args.model in classification_models else "mIoU"

    print("\n--- Bayesian Optimization Finished ---")
    print(f"Best {metric_label}: {best_metric:.4f}")
    print("Best hyperparameters:")
    for name, val in best_params.items():
        if isinstance(val, float):
            print(f"  - {name}: {val:.6f}")
        else:
            print(f"  - {name}: {val}")

    # Convert numpy types to native Python types for JSON serialization
    serializable_best_params = {}
    for name, val in best_params.items():
        if isinstance(val, np.integer):
            serializable_best_params[name] = int(val)
        elif isinstance(val, np.floating):
            serializable_best_params[name] = float(val)
        else:
            serializable_best_params[name] = val

    # Save the best hyperparameters to a JSON file
    best_results_log = {
        'best_metric': best_metric,
        'metric_name': metric_label,
        'best_hyperparameters': serializable_best_params,
    }
    with open(json_log_filename, 'w') as f:
        json.dump(best_results_log, f, indent=4)
    
    print(f"\nBest hyperparameters saved to '{json_log_filename}'")
    print(f"Full optimization log saved to '{csv_log_filename}'")

    # --- Plotting ---
    if search_space_dims:
        print("\nGenerating optimization plots...")
        try:
            import matplotlib.pyplot as plt

            plot_dir = "bayes_opt_plots"
            os.makedirs(plot_dir, exist_ok=True)

            # Plot 1: Evaluations
            _ = plot_evaluations(result, dimensions=optimizing_names)
            plt.savefig(os.path.join(plot_dir, f"evaluations_{args.model}.png"))
            plt.close()

            # Plot 2: Objective (partial dependence)
            _ = plot_objective(result, dimensions=optimizing_names)
            plt.savefig(os.path.join(plot_dir, f"objective_{args.model}.png"))
            plt.close()
            print(f"Plots saved in '{plot_dir}/'")
        except ImportError:
            print("\nWarning: Matplotlib not found. Skipping plot generation. Install with 'pip install matplotlib'")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization for segmentation or classification models."
    )
    parser.add_argument('--n-calls', type=int, default=20, help="Number of Bayesian optimization calls (trials).")
    parser.add_argument(
        '--model',
        type=str,
        default='unet_hybrid',
        choices=['unet', 'unet_hybrid', 'fcn', 'fcn_hybrid', 'jvgg19', 'jvgg21'],
        help="Model family to optimize.",
    )
    parser.add_argument('--lr', type=float, default=None, help="Learning rate. If not set, it will be optimized.")
    parser.add_argument(
        '--depth',
        type=int,
        default=None,
        help="Model depth (or stages) for segmentation models. If not set, it will be optimized.",
    )
    parser.add_argument(
        '--base-pos',
        type=int,
        default=None,
        help="Positive channels (J-Conv) or base channels. If not set, it will be optimized.",
    )
    parser.add_argument(
        '--base-neg',
        type=int,
        default=None,
        help="Negative channels (J-Conv). If not set, it will be optimized.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Batch size. If not set, it will be optimized.",
    )
    parser.add_argument('--img-size', type=int, default=256, help="Size to resize input images to.")
    parser.add_argument(
        '--gpu-mem-fraction',
        type=float,
        default=None,
        help="Fraction of GPU memory to use (e.g., 0.8 for 80%). Only for CUDA devices.",
    )
    parser.add_argument(
        '--activation',
        type=str,
        default=None,
        choices=['tanh', 'gelu', 'leaky_relu'],
        help="Activation function to use in JConv blocks.",
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help="Dropout probability for JVGG classifier head. If not set, it will be optimized.",
    )
    parser.add_argument(
        '--num-epochs-per-trial',
        type=int,
        default=10,
        help="Number of epochs to train per trial. A smaller number provides faster estimates.",
    )
    parser.add_argument('--orth', action='store_true', help="Use JConv2dOrtho instead of standard JConv.")
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help="Root path for Oxford-IIIT Pet datasets.",
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default=None,
        help="Custom folder dataset training split for classification models.",
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default=None,
        help="Custom folder dataset validation split for classification models.",
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
        help="Projection mode for JVGG models.",
    )
    parser.add_argument(
        '--avgpool-size',
        type=int,
        default=7,
        help="Adaptive average pooling output size for JVGG classifiers.",
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['in', 'out', 'output'],
        help="Orthogonal mode for JConv2dOrtho. If not set, it will be optimized.",
    )
    args = parser.parse_args()
    main(args)
