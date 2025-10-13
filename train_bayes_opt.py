import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
import csv
from skopt import gp_minimize
from skopt.plots import plot_objective, plot_evaluations
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from utils import build_unet, build_unet_hybrid_jenc, build_fcn, build_fcn_hybrid_jenc, dice_score, iou_score


def get_device():
    """Gets the best available device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


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

class PetDatasetWrapper(OxfordIIITPet):
        def __init__(self, root, split, transform=None, download=False):
            super().__init__(root=root, split=split, target_types='segmentation', download=download)
            self.transform = transform

        def __getitem__(self, index):
            img, mask = super().__getitem__(index)
            if self.transform:
                img, mask = self.transform(img, mask)
            return img, mask

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

    # Combine fixed and optimized parameters
    opt_params = {dim.name: val for dim, val in zip(search_space_dims, params)}
    all_params = {**fixed_params, **opt_params}

    # If base_pos is being optimized, ensure it's a potencie of 2
    if 'base_pos_factor' in all_params:
        all_params['base_pos'] = 2**all_params.pop('base_pos_factor')
    if 'base_neg_factor' in all_params:
        all_params['base_neg'] = 2**all_params.pop('base_neg_factor')

    # Extract params for this trial
    lr = all_params['lr']
    depth = all_params['depth']
    base_pos = all_params['base_pos']
    base_neg = all_params['base_neg']
    batch_size = all_params['batch_size']

    print(f"\n--- Bayesian Opt Trial: {trial_num} ---")
    print(f"Params: lr={lr:.6f}, depth={depth}, base_pos={base_pos}, base_neg={base_neg}, batch_size={batch_size}")

    device = get_device()

    try:
        # --- Dataset and DataLoaders ---
        dataset_transforms = PetDatasetTransforms(size=args.img_size)
        train_dataset = PetDatasetWrapper(root='./data', split='trainval', transform=dataset_transforms, download=True)
        val_dataset = PetDatasetWrapper(root='./data', split='test', transform=dataset_transforms, download=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # --- 2. Model ---
        n_classes = 3 # 0: Background, 1: Pet, 2: Border
        if args.model == 'unet':
            print(f"Building standard UNet model with base_ch={base_pos}...")
            model = build_unet(in_ch=3, n_classes=n_classes, base_ch=base_pos, depth=depth, bilinear=True)
        elif args.model == 'unet_hybrid':
            print(f"Building UNet-Hybrid model with base_pos={base_pos} and base_neg={base_neg}...")
            model = build_unet_hybrid_jenc(in_ch=3, n_classes=n_classes, base_pos=base_pos, base_neg=base_neg, depth=depth)
        elif args.model == 'fcn':
            print(f"Building standard FCN model with base_ch={base_pos}...")
            model = build_fcn(in_ch=3, n_classes=n_classes, base_ch=base_pos, stages=depth)
        elif args.model == 'fcn_hybrid':
            print(f"Building FCN-Hybrid model with base_pos={base_pos} and base_neg={base_neg}...")
            model = build_fcn_hybrid_jenc(in_ch=3, n_classes=n_classes, base_pos=base_pos, base_neg=base_neg, stages=depth)
        else:
            raise ValueError(f"Unknown model type: '{args.model}'")

        model.to(device)

        # --- Loss and Optimizer ---
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # --- Training Loop ---
        best_val_iou = -1.0
        # For optimization, we might not need to run for all epochs
        # A smaller number can give a good estimate of the hyperparameter quality
        num_epochs_per_trial = 5

        for epoch in range(num_epochs_per_trial):
            print(f"\n--- Epoch {epoch+1}/{num_epochs_per_trial} ---")
            train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            _, _, val_iou = evaluate(model, val_loader, loss_fn, n_classes, device)
            mean_iou = val_iou.mean()
            print(f"Epoch {epoch+1} - Val mIoU: {mean_iou:.4f}")

            if mean_iou > best_val_iou:
                best_val_iou = mean_iou

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"WARNING: CUDA out of memory on trial {trial_num}. Skipping.")
            torch.cuda.empty_cache()
            best_val_iou = -1.0 # Penalize this trial heavily
        else:
            raise e

    # Log trial results to CSV
    log_file = csv_log_filename
    is_new_file = not os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            header = ['trial', 'lr', 'depth', 'base_pos', 'base_neg', 'batch_size', 'best_miou']
            writer.writerow(header)
        writer.writerow([trial_num, lr, depth, base_pos, base_neg, batch_size, best_val_iou if isinstance(best_val_iou, float) else best_val_iou.item()])

    # gp_minimize tries to minimize the objective, so we return the negative of mIoU
    return -(best_val_iou if isinstance(best_val_iou, float) else best_val_iou.item())

def main(args):
    """Main function to run the Bayesian Optimization."""
    global search_space_dims, fixed_params, csv_log_filename

    # --- Define filenames based on model ---
    csv_log_filename = f'bayes_opt_log_{args.model}.csv'
    json_log_filename = f'best_hyperparameters_{args.model}.json'

    # --- Define full search space and identify fixed vs. optimized params ---
    full_search_space = {
        'lr': Real(1e-5, 1e-3, prior='log-uniform', name='lr'),
        'depth': Integer(3, 5, name='depth'),
        'base_pos': Integer(2, 7, name='base_pos_factor'), # Will be multiplied by 2
        'base_neg': Integer(2, 7, name='base_neg_factor'),
        'batch_size': Integer(4, 16, name='batch_size')
    }

    for name, space in full_search_space.items():
        arg_val = getattr(args, name)
        if arg_val is not None:
            # Special handling for base_pos if it's fixed
            if name in ('base_pos',"base_neg"):
                if arg_val % 2 != 0:
                    print(f"Warning: Provided --base-pos ({arg_val}) is not a multiple of 2. This is unusual but will be used as a fixed value.")
                fixed_params['base_pos'] = arg_val
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

    print("\n--- Bayesian Optimization Finished ---")
    print(f"Best mIoU: {-result.fun:.4f}")
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
        'best_mean_iou': -result.fun,
        'best_hyperparameters': serializable_best_params
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
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization for segmentation models.")
    parser.add_argument('--n-calls', type=int, default=20,
                        help="Number of Bayesian optimization calls (trials).")
    parser.add_argument('--model', type=str, default='unet_hybrid', choices=['unet', 'unet_hybrid', 'fcn', 'fcn_hybrid'],
                        help="Model to train ('unet', 'unet_hybrid', 'fcn', 'fcn_hybrid').")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate. If not set, it will be optimized.")
    parser.add_argument('--depth', type=int, default=None, help="Model depth (or stages for FCN). If not set, it will be optimized.")
    parser.add_argument('--base-pos', type=int, default=None, help="Positive channels (J-Conv) or base channels (standard conv). If not set, it will be optimized.")
    parser.add_argument('--base-neg', type=int, default=None, help="Negative channels (J-Conv). If not set, it will be optimized.")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size. If not set, it will be optimized.")
    parser.add_argument('--img-size', type=int, default=256,
                        help="Size to resize input images to.")
    parser.add_argument('--gpu-mem-fraction', type=float, default=None,
                        help="Fraction of GPU memory to use (e.g., 0.8 for 80%). Only for CUDA devices.")

    args = parser.parse_args()
    main(args)