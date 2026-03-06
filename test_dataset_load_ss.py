import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
import argparse
import json
import os
from sklearn.metrics import accuracy_score, f1_score

# Importamos tus módulos personalizados
from dataset import (
    PetDatasetTransforms,
    PetDatasetWrapper,
    PetClassificationTransforms,
    PetClassificationWrapper,
    FolderDatasetTransforms,
    FolderDatasetWrapper,
    YOLOSegDataset
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

# --- FUNCIONES DE EVALUACIÓN (Versiones Corregidas) ---

def evaluate_seg(model, loader, loss_fn, n_classes, device, wrapper_folder=False):
    """Evaluates the model on the test set for Segmentation."""
    model.eval()
    total_loss = 0.0
    
    # Inicializadores seguros
    total_dice = torch.zeros(n_classes, device=device) if n_classes > 1 else 0.0
    total_iou = torch.zeros(n_classes, device=device) if n_classes > 1 else 0.0
    
    with torch.no_grad():
        loop = tqdm(loader, desc="Testing")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            # --- LÓGICA DE CORRECCIÓN DE DIMENSIONES ---
            if wrapper_folder:
                # Si viene con canales extra (Batch, 2, H, W), colapsamos a indices (Batch, H, W)
                if masks.ndim == 4:
                    masks = torch.argmax(masks, dim=1)
                masks = masks.long() 
            else:
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                masks = masks.long()
            # -------------------------------------------

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

def evaluate_cls(model, loader, loss_fn, device, score='f1'):
    """Evaluates the model on the test set for Classification."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        loop = tqdm(loader, desc="Testing")
        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)

    if score == 'accuracy':
        metric_value = accuracy_score(all_targets, all_preds)
    elif score == 'f1':
        metric_value = f1_score(all_targets, all_preds, average='macro')
    else:
        metric_value = 0.0

    return avg_loss, metric_value

# --- MAIN TESTING LOGIC ---

def test(args):
    device = get_device()
    print(f"Using device: {device}")

    # 1. Cargar configuración desde el JSON
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Si el JSON guardó todo en "best_hyperparameters", úsalo, si no, usa el root
    params = config.get('best_hyperparameters', config) if 'best_hyperparameters' in config else config
    
    # Imprimir configuración cargada
    print("\n--- Loaded Configuration ---")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    # 2. Reconstruir parámetros necesarios (Prioridad: Argumentos > JSON > Default)
    model_name = args.model if args.model else params.get('model', 'unet')
    img_size = args.img_size
    
    # Parámetros específicos del modelo
    base_pos = params.get('base_pos', 32)
    base_neg = params.get('base_neg', 8)
    depth = params.get('depth', 4)
    activation = params.get('activation', 'relu')
    orth = params.get('orth', False)
    
    # Parámetros Dataset Segmentation
    dataset_classes = args.classes if args.classes else params.get('classes', None)
    
    # <--- CORRECCIÓN AQUÍ: Leemos add_background de args o params
    add_background = args.add_background or params.get('add_background', False)
    
    is_wrapper_folder = args.wrapper_folder # Esto suele pasarse por flag
    
    # Sobrescribir data_root si se pasa por consola
    data_root = args.data_root if args.data_root else params.get('data_root', './data')

    classification_models = {'jvgg19', 'jvgg21','vgg19','vgg21'}
    is_classification = model_name in classification_models

    # --- 3. Dataset Setup (Apuntando a TEST) ---
    print("\n--- Preparing Test Data ---")
    
    if is_classification:
        cls_transform = PetClassificationTransforms(size=img_size)
        if args.test_dir:
             transform = FolderDatasetTransforms(size=img_size)
             test_dataset = FolderDatasetWrapper(args.test_dir, transform=transform)
             n_classes = len(test_dataset.classes)
        else:
             test_dataset = PetClassificationWrapper(root=data_root, split='test', transform=cls_transform, download=True)
             n_classes = len(test_dataset.dataset.classes)
    else:
        # Lógica de Segmentación
        if is_wrapper_folder:
            print(f"Loading YOLOSegDataset from: {os.path.join(data_root, 'test')}")
            
            test_transforms = v2.Compose([
                v2.Resize((img_size, img_size)),
                v2.ToDtype(torch.float32, scale=True),
            ])
            
            # NOTA: Aquí buscamos en 'test/images' y 'test/labels'
            test_dataset = YOLOSegDataset(
                images_dir=os.path.join(data_root, 'test/images'),
                labels_dir=os.path.join(data_root, 'test/labels'),
                transform=test_transforms,
                classes=dataset_classes,
                add_background=add_background, # <--- Se usa la variable corregida
                target_class_id=args.target_class_id
            )
            # Recalcular n_classes basado en la lógica del dataset
            n_classes = test_dataset.num_classes
        else:
            dataset_transforms = PetDatasetTransforms(size=img_size)
            test_dataset = PetDatasetWrapper(root=data_root, split='test', transform=dataset_transforms, download=True)
            n_classes = 3 # Default Pet dataset
    
    # Sobrescribir num_classes si se fuerza
    if args.num_classes is not None:
        n_classes = args.num_classes

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes (Output Channels): {n_classes}")

    # --- 4. Model Building ---
    print(f"\n--- Building Model: {model_name} ---")
    
    if model_name == 'unet':
        model = build_unet(in_ch=3, n_classes=n_classes, base_ch=base_pos, depth=depth, bilinear=True)
    elif model_name == 'unet_hybrid':
        model = build_unet_hybrid_jenc(in_ch=3, n_classes=n_classes, base_pos=base_pos, base_neg=base_neg, depth=depth, act=activation, orth=orth)
    elif model_name == 'fcn':
        model = build_fcn(in_ch=3, n_classes=n_classes, base_ch=base_pos, stages=depth)
    elif model_name == 'fcn_hybrid':
        model = build_fcn_hybrid_jenc(in_ch=3, n_classes=n_classes, base_pos=base_pos, base_neg=base_neg, stages=depth, act=activation)
    elif model_name in classification_models:
        if 'jvgg19' in model_name:
             model = build_jvgg19(in_ch=3, base_pos=base_pos, base_neg=base_neg, n_classes=n_classes, act=activation, orth=orth, proj_mode=params.get('proj_mode','sub'), avgpool_size=params.get('avgpool_size',7), dropout=params.get('dropout',0.5))
        # Agrega aquí los otros modelos si es necesario...
        else:
             raise ValueError(f"Model implementation for {model_name} needed in test.py")
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    model.to(device)

    # --- 5. Load Weights ---
    print(f"Loading weights from: {args.weights}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights file not found: {args.weights}")
        
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    print("Weights loaded successfully.")

    # --- 6. Run Evaluation ---
    loss_fn = nn.CrossEntropyLoss()
    
    if is_classification:
        test_loss, test_metric = evaluate_cls(model, test_loader, loss_fn, device, score='accuracy')
        print(f"\nResults:\n  Test Loss: {test_loss:.4f}\n  Test Accuracy: {test_metric*100:.2f}%")
    else:
        # SEGMENTATION EVALUATION
        test_loss, test_dice, test_iou = evaluate_seg(
            model, 
            test_loader, 
            loss_fn, 
            n_classes, 
            device, 
            wrapper_folder=is_wrapper_folder 
        )
        
        print(f"\n--- Test Results (Segmentation) ---")
        print(f"Average Loss: {test_loss:.4f}")
        
        # Generar nombres de clases si existen
        cls_names = dataset_classes if dataset_classes else [str(i) for i in range(n_classes)]
        
        # Ajuste visual para nombres si se agregó fondo y no está en la lista original
        if add_background and len(cls_names) == n_classes - 1:
            cls_names = ['Background'] + list(cls_names)
        elif add_background and len(cls_names) == n_classes:
            # Asumimos que el usuario ya sabe que el 0 es fondo
            pass 
            
        print("\nPer-Class Metrics:")
        for i in range(n_classes):
            c_name = str(cls_names[i]) if i < len(cls_names) else f"Class {i}"
            print(f"  {c_name}:")
            print(f"    Dice: {test_dice[i]:.4f}")
            print(f"    IoU:  {test_iou[i]:.4f}")

        print(f"\nMean IoU (mIoU): {test_iou.mean().item():.4f}")
        print(f"Mean Dice:     {test_dice.mean().item():.4f}")
        print("-----------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test dataset.")
    
    # Argumentos obligatorios
    parser.add_argument('--config', type=str, required=True, help="Path to the .json config file.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the .pth model weights file.")
    
    # Argumentos opcionales de Override
    parser.add_argument('--data-root', type=str, default=None, help="Root path for data.")
    parser.add_argument('--test-dir', type=str, default=None, help="Specific path to test folder (classification).")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for testing.")
    
    # Flags importantes para tu caso
    parser.add_argument('--wrapper-folder', action='store_true', help="Use YOLOSegDataset.")
    
    # <--- AQUÍ ESTÁ EL QUE FALTABA
    parser.add_argument('--add-background', action='store_true', help="Force add background class.")
    
    parser.add_argument('--classes', nargs='+', type=int, default=None, help="Override classes list.")
    parser.add_argument('--model', type=str, default=None, help="Force model name.")
    parser.add_argument('--num-classes', type=int, default=None, help="Force num classes.")
    parser.add_argument('--img-size', type=int, default=256, help="Force image size.")
    parser.add_argument("--target-class-id", type=int, default=None, help="Target class ID for binary segmentation.")
    
    args = parser.parse_args()
    test(args)