# /home/craljimenez/Documents/PhD/001_JCONV/compute_clscam.py
import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Dict, List

# Reutilizamos funciones de nuestros otros scripts
from utils import (
    build_jvgg19, build_jvgg21, build_vgg19, build_vgg21, load_dataset
)
from compute_segcam import denormalize, get_module_by_path

# --- Definición de la nueva clase ClassificationCAM ---
class ClassificationCAM:
    """
    Clase para calcular CAM (Grad-CAM, Grad-CAM++, Score-CAM) para modelos de clasificación.
    Adaptada de SegmentationCAM.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, mode: str = "gradcam"):
        self.model = model
        self.target_layer = target_layer
        self.mode = mode.lower()
        if self.mode not in {"gradcam", "gradcam++", "scorecam"}:
            raise ValueError(f"Modo CAM desconocido '{mode}'.")

        self.activations = None
        self.gradients = None
        self.multi_branch = False
        self._suppress_hooks = False
        self._register_hooks()

    def _unpack_grad_output(self, grad_output):
        """
        Desempaqueta de forma segura la tupla de gradientes que devuelve el hook.
        PyTorch puede anidar los gradientes en una tupla extra, ej: ((grad1, grad2),).
        """
        if isinstance(grad_output, tuple) and len(grad_output) == 1:
            return grad_output[0]
        return grad_output

    def _register_hooks(self):
        def forward_hook(module, _inp, output):
            if self._suppress_hooks: return
            # Siempre almacenar como una tupla
            self.activations = output if isinstance(output, tuple) else (output,)
            self.multi_branch = isinstance(output, (tuple, list)) and len(output) > 1

        def backward_hook(module, grad_input, grad_output):
            if self._suppress_hooks: return
            # Usar la función de desempaquetado para manejar la estructura de gradientes
            unpacked_grads = self._unpack_grad_output(grad_output)
            self.gradients = unpacked_grads if isinstance(unpacked_grads, tuple) else (unpacked_grads,)

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove(self):
        if self.forward_handle: self.forward_handle.remove()
        if self.backward_handle: self.backward_handle.remove()

    def _compute_grad_based_cam(self, inputs: torch.Tensor, outputs: torch.Tensor, class_ids: List[int]):
        cams_per_class: Dict[int, Dict[str, torch.Tensor] | torch.Tensor] = {}
        for idx, class_id in enumerate(class_ids):
            self.model.zero_grad(set_to_none=True)
            score = outputs[:, class_id]

            retain = idx < len(class_ids) - 1
            score.backward(retain_graph=retain)

            if self.gradients is None or not any(g is not None for g in self.gradients) or len(self.gradients) != len(self.activations):
                print(f"  [Debug] No se capturaron gradientes para la clase {class_id}. Saltando.")
                continue

            branch_maps = {}
            for branch_idx, (activation, gradient) in enumerate(zip(self.activations, self.gradients)):
                if activation is None or gradient is None: continue

                activation = activation.detach()
                gradient = gradient.detach()

                weights = gradient.mean(dim=(2, 3), keepdim=True) # Grad-CAM
                if self.mode == "gradcam++":
                    grad2 = gradient.pow(2)
                    grad3 = grad2 * gradient
                    denom = 2 * grad2 + (activation * grad3).sum(dim=(2, 3), keepdim=True)
                    alpha = grad2 / (denom + 1e-8)
                    weights = (alpha * torch.relu(gradient)).sum(dim=(2, 3), keepdim=True)

                # Mantener la dimensión del batch para interpolate
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                # Interpolar primero, luego exprimir (squeeze)
                cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
                cam = cam.squeeze()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                if self.multi_branch:
                    label = "pos" if branch_idx == 0 else "neg"
                    branch_maps[label] = cam.cpu()
                else:
                    branch_maps["cam"] = cam.cpu()

            if branch_maps:
                cams_per_class[class_id] = branch_maps

        return cams_per_class

    def generate(self, inputs: torch.Tensor, class_ids: List[int]):
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)

        if self.mode in ("gradcam", "gradcam++"):
            return outputs, self._compute_grad_based_cam(inputs, outputs, class_ids)
        else:
            raise NotImplementedError("Score-CAM para clasificación aún no implementado en este script.")

def resolve_classification_target_layer(model, model_type: str, override: Optional[str] = None):
    """Resuelve la capa objetivo para modelos de clasificación."""
    if override:
        return get_module_by_path(model, override)
    if 'jvgg' in model_type:
        # La última etapa convolucional antes de la proyección es un buen objetivo
        return model.stages[-1]
    if 'vgg' in model_type:
        # En VGG, es la última secuencia de capas convolucionales
        return model.features[-1]
    raise ValueError(f"Resolución de capa CAM no implementada para el modelo '{model_type}'.")

def filter_config_for_model(config: dict, expected_args: list) -> dict:
    """
    Filtra el diccionario de configuración para que solo contenga los argumentos
    esperados por una función constructora de modelos.
    """
    model_params = {}
    for key, value in config.items():
        # El argumento 'base_pos' en VGG se mapea a 'base_ch'
        if key == 'base_pos' and 'base_ch' in expected_args:
            model_params['base_ch'] = value
        elif key == "activation" and 'act' in expected_args:
            model_params['act'] = value
        elif key in expected_args:
            model_params[key] = value
    return model_params

def load_hyperparameters(json_path: Path) -> dict:
    """
    Carga los hiperparámetros desde un archivo JSON, esperando la clave 'best_hyperparameters'.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("best_hyperparameters", {})


def main():
    """
    python compute_clscam.py \
    --config-path models/val_loader_config_jvgg19.json \
    --index 15 \
    --cam-type gradcam
    """
    parser = argparse.ArgumentParser(description="Compute and visualize CAM for classification models.")
    parser.add_argument("--config-path", type=Path, required=True, help="Path to the val_loader_config JSON file.")
    parser.add_argument("--index", type=int, required=True, help="Dataset index to inspect.")
    parser.add_argument("--target-class", type=int, default=None, help="Class ID for CAM. Defaults to the predicted class.")
    parser.add_argument("--cam-type", type=str, default="gradcam", choices=["gradcam", "gradcam++"], help="Type of CAM to compute.")
    parser.add_argument("--all-classes", action="store_true", help="Generate CAMs for all classes instead of just one.")
    parser.add_argument("--target-layer-name", type=str, default=None, help="Dotted path to the target module for CAM (e.g., 'stages[-1]').")
    parser.add_argument("--output-dir", type=Path, default=Path("./clscam_outputs"), help="Directory to store visualizations.")
    parser.add_argument("--hyperparameters", type=Path, default=None, help="Path to a JSON file with model hyperparameters (e.g., from Bayesian optimization).")
    parser.add_argument("--best-model-params",type=str,default=None,help="Best params to model")
    args = parser.parse_args()

    # --- 1. Cargar configuración y recrear DataLoader ---
    val_loader, config, device = load_dataset(args.config_path)
    model_name = config['model']
    n_classes = config['num_classes']
    if args.best_model_params:
        best_model_path = args.best_model_params
    else:
        best_model_path = config['best_model_path']

    # --- Cargar hiperparámetros adicionales si se proporcionan ---
    if args.hyperparameters:
        print(f"Loading hyperparameters from: {args.hyperparameters}")
        hyper_params = load_hyperparameters(args.hyperparameters)
        config.update(hyper_params) # Los nuevos hiperparámetros sobreescriben los del config
        print("Hyperparameters updated.")

    # --- 2. Reconstruir y cargar el modelo ---
    print(f"\nRebuilding model: '{model_name}'...")
    if model_name == 'jvgg19':
        expected = ['base_pos', 'base_neg', 'act', 'orth', 'proj_mode', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(config, expected)
        model = build_jvgg19(n_classes=n_classes, **model_kwargs)
    elif model_name == 'jvgg21':
        expected = ['base_pos', 'base_neg', 'act', 'orth', 'proj_mode', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(config, expected)
        model = build_jvgg21(n_classes=n_classes, **model_kwargs)
    elif model_name == 'vgg19':
        # build_vgg19 espera 'base_ch', no 'base_pos'
        expected = ['base_ch', 'base_pos', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(config, expected)
        model = build_vgg19(n_classes=n_classes, **model_kwargs)
    elif model_name == 'vgg21':
        # build_vgg21 espera 'base_ch', no 'base_pos'
        expected = ['base_ch', 'base_pos', 'avgpool_size', 'dropout']
        model_kwargs = filter_config_for_model(config, expected)
        model = build_vgg21(n_classes=n_classes, **model_kwargs)
    else:
        raise ValueError(f"Model type '{model_name}' is not a supported classification model.")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 3. Obtener la imagen y la etiqueta ---
    if args.index >= len(val_loader.dataset):
        raise IndexError(f"Index {args.index} is out of range for the dataset with {len(val_loader.dataset)} samples.")
    
    image, label = val_loader.dataset[args.index]
    image_batch = image.unsqueeze(0).to(device)

    # --- 4. Calcular CAM ---
    target_layer = resolve_classification_target_layer(model, model_name, override=args.target_layer_name)
    print(f"Using target layer '{args.target_layer_name or 'default'}' ({target_layer.__class__.__name__}) for {args.cam_type}.")

    cam_helper = ClassificationCAM(model, target_layer, mode=args.cam_type)

    # Determinar para qué clases generar el CAM
    with torch.no_grad():
        outputs = model(image_batch)
    predicted_class = outputs.argmax(dim=1).item()

    if args.all_classes:
        class_ids_to_process = list(range(n_classes))
        print(f"Generando CAMs para todas las clases: {class_ids_to_process}")
    else:
        class_ids_to_process = [args.target_class if args.target_class is not None else predicted_class]

    with torch.enable_grad():
        outputs, cam_dict = cam_helper.generate(image_batch, class_ids=class_ids_to_process)
    cam_helper.remove()

    # --- 5. Visualizar Resultados ---
    image_np = denormalize(image).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    cam_entries = []
    for class_id, class_maps in cam_dict.items():
        if not class_maps: continue
        # La lógica de visualización ahora maneja correctamente ambos casos
        if isinstance(class_maps, dict):
            for branch, cam_tensor in class_maps.items():
                title = f"{args.cam_type.title()} (Cls {class_id}, Rama {branch})" if branch in ['pos', 'neg'] else f"{args.cam_type.title()} (Cls {class_id})"
                cam_entries.append((title, cam_tensor))
        else:
            # Fallback por si la estructura cambia
            cam_entries.append((f"{args.cam_type.title()} (Cls {class_id})", class_maps))

    if not cam_entries:
        print(f"ADVERTENCIA: No se generaron mapas CAM para las clases solicitadas. La imagen de salida no tendrá superposición.")

    total_cols = 1 + len(cam_entries)
    fig, axes = plt.subplots(1, total_cols, figsize=(6 * total_cols, 6))
    if not hasattr(axes, '__len__'): axes = [axes]

    axes[0].imshow(image_np)
    axes[0].set_title(f"Input\nTrue: {label} | Pred: {predicted_class}")
    
    if cam_entries:
        for i, (title, cam_tensor) in enumerate(cam_entries, 1):
            cam_np = cam_tensor.numpy()
            axes[i].imshow(image_np)
            axes[i].imshow(cam_np, cmap="jet", alpha=0.5)
            axes[i].set_title(title)

    for ax in axes: ax.axis("off")
    fig.tight_layout()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"clscam_{model_name}_{args.cam_type}_idx{args.index}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    main()
