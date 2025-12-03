# ...existing code...
import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Optional, Sequence, Dict, List

# Reutilizamos funciones de nuestros otros scripts
from utils import (
    build_jvgg19, build_jvgg21, build_vgg19, build_vgg21, load_dataset
)
from compute_segcam import denormalize, get_module_by_path


from cls_cams import (ClassificationCAM
                      ,load_hyperparameters
                      ,filter_config_for_model
                      ,resolve_classification_target_layer
                      )


def _extract_state_dict_from_checkpoint(ckpt):
    """
    Return a plain state_dict from various checkpoint layouts.
    Handles:
      - direct state_dict
      - {'state_dict': ...} or {'model_state_dict': ...}
      - DataParallel prefixes 'module.' (stripped)
    """
    if isinstance(ckpt, dict):
        # common wrappers
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            sd = ckpt['state_dict']
        elif 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt
    else:
        sd = ckpt

    # If keys have 'module.' prefix (DataParallel), strip it
    new_sd = {}
    for k, v in sd.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        new_sd[new_key] = v
    return new_sd


def _infer_bases_from_state_dict(sd: Dict[str, torch.Tensor]):
    """
    Try to infer base_pos and base_neg from checkpoint parameter shapes.
    Returns tuple (base_pos, base_neg) where each may be None.
    Heuristics:
      - Look for keys like 'lift.lift_neg.weight' or 'lift_neg.weight' etc.
      - Fallback: search for first conv negative/positive parameter naming patterns.
    """
    base_pos = None
    base_neg = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            if k.endswith('lift.lift_neg.weight') or k.endswith('lift_neg.weight') or '.lift_neg.weight' in k:
                base_neg = v.shape[0]
            if k.endswith('lift.lift_pos.weight') or k.endswith('lift_pos.weight') or '.lift_pos.weight' in k:
                base_pos = v.shape[0]
            # also attempt to spot first conv layer negative/positive channels
            if '.stages.0.layers.0.conv.conv_neg.weight' in k or '.stages.0.layers.0.conv.R_param' in k:
                # Many architectures store channel count as out_channels for that tensor
                try:
                    base_neg = v.shape[0]
                except Exception:
                    pass
            if '.stages.0.layers.0.conv.conv_pos.weight' in k:
                try:
                    base_pos = v.shape[0]
                except Exception:
                    pass
    return base_pos, base_neg


def _get_orth_from_state_dict(sd: Dict[str, torch.Tensor]) -> Optional[bool]:
    """Infiere si el modelo es ortogonal buscando claves 'R_param'."""
    return any('.R_param' in k for k in sd.keys())


def _build_model_by_name(model_name, n_classes, final_config):
    """
    Centralize model construction to allow rebuilding when we infer different params.
    """
    if model_name == 'jvgg19':
        expected = ['base_pos', 'base_neg', 'act', 'orth', 'proj_mode', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(final_config, expected)
        print("model_kwargs:", model_kwargs)
        return build_jvgg19(n_classes=n_classes, **model_kwargs)
    elif model_name == 'jvgg21':
        expected = ['base_pos', 'base_neg', 'act', 'orth', 'proj_mode', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(final_config, expected)
        return build_jvgg21(n_classes=n_classes, **model_kwargs)
    elif model_name == 'vgg19':
        expected = ['base_ch', 'base_pos', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(final_config, expected)
        return build_vgg19(n_classes=n_classes, **model_kwargs)
    elif model_name == 'vgg21':
        expected = ['base_ch', 'base_pos', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(final_config, expected)
        return build_vgg21(n_classes=n_classes, **model_kwargs)
    else:
        raise ValueError(f"Model type '{model_name}' is not a supported classification model.")


def main():
    parser = argparse.ArgumentParser(description="Compute and visualize CAM for classification models.")
    parser.add_argument("--config-path", type=Path, required=True, help="Path to the val_loader_config JSON file.")
    parser.add_argument("--target-class", type=int, default=None, help="Class ID for CAM. Defaults to the predicted class.")
    parser.add_argument("--cam-type", type=str, default="gradcam", choices=["gradcam", "gradcam++"], help="Type of CAM to compute.")
    parser.add_argument("--all-classes", action="store_true", help="Generate CAMs for all classes instead of just one.")
    parser.add_argument("--target-layer-name", type=str, default=None, help="Dotted path to the target module for CAM (e.g., 'stages[-1]').")
    parser.add_argument("--output-dir", type=Path, default=Path("./clscam_outputs"), help="Directory to store visualizations.")
    parser.add_argument("--hyperparameters", type=Path, default=None, help="Path to a JSON file with model hyperparameters (e.g., from Bayesian optimization).")
    parser.add_argument("--best-model-params",type=str,default=None,help="Best params to model")
    parser.add_argument("--output-csv", type=Path, default=None, help="Path to save the evaluation results in a CSV file.")
    args = parser.parse_args()

    # --- Cargar hiperparámetros adicionales si se proporcionan ---
    if args.hyperparameters:
        print(f"Loading hyperparameters from: {args.hyperparameters}")
        hyper_params = load_hyperparameters(args.hyperparameters)
    else:
        hyper_params = {}

    # --- 1. Cargar configuración y recrear DataLoader ---
    val_loader, config, device = load_dataset(args.config_path)
    model_name = config['model']

    # --- Imprimir el mapeo de clases ---
    print("\n--- Mapeo de Clases (Índice -> Nombre) ---")
    # Acceder al dataset original, incluso si es un subconjunto (Subset)
    dataset_obj = val_loader.dataset
    if hasattr(dataset_obj, 'dataset'):
        dataset_obj = dataset_obj.dataset

    if hasattr(dataset_obj, 'class_to_idx'):
        class_to_idx = dataset_obj.class_to_idx
        # Invertir el diccionario para tener {índice: nombre}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        for idx in sorted(idx_to_class.keys()):
            print(f"  - Target {idx}: {idx_to_class[idx]}")
    else:
        print("  - No se pudo encontrar el mapeo de clases en el dataset.")

    n_classes = config['num_classes']
    if args.best_model_params:
        best_model_path = args.best_model_params
    else:
        best_model_path = config['best_model_path']

    # Combinar configuraciones: los hiperparámetros del JSON tienen prioridad.
    final_config = {**config, **hyper_params}
    if hyper_params:
        print("Hyperparameters updated.")

    # --- 2. Reconstruir y cargar el modelo ---
    print(f"\nRebuilding model: '{model_name}'...")
    # initial build using final_config
    model = _build_model_by_name(model_name, n_classes, final_config)

    # Load checkpoint and handle mismatches gracefully
    raw_ckpt = torch.load(best_model_path, map_location=device)
    state_dict = _extract_state_dict_from_checkpoint(raw_ckpt)

    # Inferir 'orth' directamente del state_dict, ya que es la causa más probable de error.
    is_orth_in_ckpt = _get_orth_from_state_dict(state_dict)
    if is_orth_in_ckpt and not final_config.get('orth'):
        print("Checkpoint parece ser ortogonal ('R_param' encontrado). Forzando 'orth=True' para la reconstrucción.")
        final_config['orth'] = True
        # Reconstruir el modelo con la configuración corregida
        model = _build_model_by_name(model_name, n_classes, final_config)

    # Intentar cargar el state_dict.
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Checkpoint loaded successfully (strict mode).")
    except RuntimeError as e:
        print("\nERROR: La carga estricta del state_dict falló. Esto indica una discrepancia persistente en la arquitectura.")
        print("Detalles del error original:", str(e))
        raise RuntimeError("No se pudo cargar el checkpoint del modelo. Asegúrate de que los hiperparámetros coincidan con los del entrenamiento.") from e

    model.to(device)
    model.eval()

    # --- 3. Obtener la imagen y la etiqueta ---
    # --- 4. Calcular CAM ---
    target_layer = resolve_classification_target_layer(model, model_name, override=args.target_layer_name)
    print(f"Using target layer '{args.target_layer_name or 'default'}' ({target_layer.__class__.__name__}) for {args.cam_type}.")

    cam_helper = ClassificationCAM(model, target_layer, mode=args.cam_type)

    all_preds = []
    all_targets = []
    all_original_confidences = []
    all_explainable_confidences_combined = []
    all_explainable_confidences_pos = []
    all_explainable_confidences_neg = []

    for index in range(len(val_loader.dataset)):
        image, label = val_loader.dataset[index]
        image_batch = image.unsqueeze(0).to(device)

        # Determinar para qué clases generar el CAM
        with torch.no_grad():
            outputs = model(image_batch)
        predicted_class = outputs.argmax(dim=1).item()

        class_ids_to_process = [args.target_class if args.target_class is not None else predicted_class]

        with torch.enable_grad():
            outputs, cam_dict = cam_helper.generate(image_batch, class_ids=class_ids_to_process)
        cam_helper.remove()

        # --- 5. Visualizar Resultados ---
        image_np = denormalize(image).permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # Calculate original confidence
        original_confidence = torch.softmax(outputs, dim=1)[0, label].item()

        # Apply CAM and calculate explainable confidence
        cam_maps = list(cam_dict.values())[0] if cam_dict else {}
        
        explainable_confidence_combined = None
        explainable_confidence_pos = None
        explainable_confidence_neg = None

        # Manejar modelos de una o dos ramas (J-Conv)
        if 'cam' in cam_maps:
            # Modelo estándar de una rama
            cam_tensor = cam_maps['cam']
            if cam_tensor is not None:
                cam_np = cam_tensor.numpy()
                explainable_image = image_np * cam_np[..., None]
                explainable_image_tensor = torch.tensor(explainable_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    explainable_output = model(explainable_image_tensor)
                explainable_confidence_combined = torch.softmax(explainable_output, dim=1)[0, label].item()

        elif 'pos' in cam_maps and 'neg' in cam_maps:
            # Modelo J-Conv de dos ramas: calcular métricas para cada rama
            print(f"  [Info] Calculando métricas CAM para ramas 'pos' y 'neg' en la imagen {index}.")
            
            # Rama Positiva
            cam_pos_tensor = cam_maps.get('pos')
            if cam_pos_tensor is not None:
                cam_pos_np = cam_pos_tensor.numpy()
                explainable_image_pos = image_np * cam_pos_np[..., None]
                explainable_image_pos_tensor = torch.tensor(explainable_image_pos).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    explainable_output_pos = model(explainable_image_pos_tensor)
                explainable_confidence_pos = torch.softmax(explainable_output_pos, dim=1)[0, label].item()

            # Rama Negativa
            cam_neg_tensor = cam_maps.get('neg')
            if cam_neg_tensor is not None:
                cam_neg_np = cam_neg_tensor.numpy()
                explainable_image_neg = image_np * cam_neg_np[..., None]
                explainable_image_neg_tensor = torch.tensor(explainable_image_neg).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    explainable_output_neg = model(explainable_image_neg_tensor)
                explainable_confidence_neg = torch.softmax(explainable_output_neg, dim=1)[0, label].item()

        all_preds.append(predicted_class)
        all_targets.append(label)
        all_original_confidences.append(original_confidence)
        all_explainable_confidences_combined.append(explainable_confidence_combined)
        all_explainable_confidences_pos.append(explainable_confidence_pos)
        all_explainable_confidences_neg.append(explainable_confidence_neg)

    # --- 6. Guardar resultados en un archivo CSV ---
    if args.output_csv:
        results_data = {
            'image_index': list(range(len(val_loader.dataset))),
            'true_label': all_targets,
            'predicted_label': all_preds,
            'original_confidence': all_original_confidences,
            'explainable_confidence': all_explainable_confidences_combined,
            'explainable_confidence_pos': all_explainable_confidences_pos,
            'explainable_confidence_neg': all_explainable_confidences_neg,
        }
        df = pd.DataFrame(results_data)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nEvaluation results saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
# ...existing code...