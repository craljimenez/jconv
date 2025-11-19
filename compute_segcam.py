import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Dict, List

from dataset import PetDatasetTransforms, PetDatasetWrapper
from utils import build_unet, build_unet_hybrid_jenc, build_fcn, build_fcn_hybrid_jenc, dice_score, iou_score


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_hyperparameters(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("best_hyperparameters", {})


def get_module_by_path(root: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    """
    Locate a submodule using a dotted path, allowing numeric indices for ModuleList/Sequential.
    """
    current = root
    # Regex to match 'name[index]' pattern, e.g., 'encoder[-1]'
    pattern = re.compile(r"(\w+)\[(-?\d+)\]")

    for part in layer_path.split("."):
        if not part:
            continue

        match = pattern.match(part)
        if match:
            name, index_str = match.groups()
            idx = int(index_str)
            if not hasattr(current, name):
                raise ValueError(f"Layer path '{layer_path}' is invalid: module '{name}' not found.")
            current = getattr(current, name)
            if not isinstance(current, (nn.ModuleList, nn.Sequential)):
                raise ValueError(f"Cannot index into module '{type(current).__name__}' using segment '{part}'.")
            try:
                current = current[idx]
            except IndexError as exc:
                raise ValueError(f"Index {idx} out of range for module list in '{layer_path}'.") from exc
        else:
            # Handle named attributes (e.g., 'block', 'conv_pos') and numeric indices for Sequential/ModuleList.
            if hasattr(current, part):
                candidate = getattr(current, part)
                if isinstance(candidate, nn.Module):
                    current = candidate
                    continue
            # Fallback for numeric indices in ModuleList/Sequential not caught by regex
            try:
                current = current[int(part)]
            except (ValueError, IndexError, TypeError) as exc:
                raise ValueError(f"Layer path '{layer_path}' is invalid at segment '{part}'.") from exc

    if not isinstance(current, nn.Module):
        raise ValueError(f"Resolved object for '{layer_path}' is not a torch.nn.Module.")
    return current


def resolve_target_layer(model, model_type: str, override: Optional[str] = None):
    if override:
        return get_module_by_path(model, override)
    if model_type == "unet":
        return model.ups[-1].conv.block
    if model_type == "unet_hybrid":
        return model.dec_convs[-1]
    if model_type == "fcn":
        return model.encoder[-1]
    if model_type == "fcn_hybrid":
        return model.encoder[-1]
    raise ValueError(f"Seg-CAM layer resolution not implemented for model '{model_type}'.")


class SegmentationCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, mode: str = "gradcam", scorecam_topk: int = 32):
        self.model = model
        self.target_layer = target_layer
        self.mode = mode.lower()
        if self.mode not in {"gradcam", "gradcam++", "scorecam"}:
            raise ValueError(f"Unknown CAM mode '{mode}'. Choose from 'gradcam', 'gradcam++', 'scorecam'.")
        self.scorecam_topk = scorecam_topk
        self.activations = None
        self.gradients = None
        self.multi_branch = False
        self._suppress_hooks = False
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, _inp, output):
            if self._suppress_hooks:
                return
            if isinstance(output, (tuple, list)):
                self.activations = tuple(output)
                self.multi_branch = len(self.activations) > 1
            else:
                self.activations = (output,)
                self.multi_branch = False

        def backward_hook(module, grad_input, grad_output):
            if self._suppress_hooks:
                return
            grads = grad_output if isinstance(grad_output, (tuple, list)) else (grad_output,)
            self.gradients = tuple(grads)

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _validate_single_sample(self, tensor: torch.Tensor, name: str):
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} currently supports batch size 1; got {tensor.shape[0]}.")

    def _compute_grad_based_cam(self, inputs: torch.Tensor, outputs: torch.Tensor, class_ids: List[int]):
        cams_per_class: Dict[int, Dict[str, torch.Tensor] | torch.Tensor] = {}
        for idx, class_id in enumerate(class_ids):
            self.model.zero_grad(set_to_none=True)
            score = outputs[:, class_id, :, :].mean()
            retain = idx < len(class_ids) - 1
            score.backward(retain_graph=retain)

            branch_maps = {}
            for branch_idx, (activation, gradient) in enumerate(zip(self.activations, self.gradients)):
                if activation is None or gradient is None:
                    continue
                activation = activation.detach()
                gradient = gradient.detach()
                if self.mode == "gradcam++":
                    grad2 = gradient * gradient
                    grad3 = grad2 * gradient
                    eps = 1e-8
                    denom = 2.0 * grad2 + (activation * grad3).sum(dim=(2, 3), keepdim=True)
                    denom = denom + eps
                    alpha = grad2 / denom
                    weights = (alpha * torch.relu(gradient)).sum(dim=(2, 3), keepdim=True)
                else:
                    weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
                cam = cam.squeeze()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                if self.multi_branch:
                    label = "pos" if branch_idx == 0 else ("neg" if branch_idx == 1 else f"branch_{branch_idx}")
                    branch_maps[label] = cam.cpu()
                else:
                    branch_maps["cam"] = cam.cpu()

            if not branch_maps:
                raise RuntimeError("Grad-CAM computation failed: empty CAM map.")

            if self.multi_branch:
                cams_per_class[class_id] = branch_maps
            else:
                cams_per_class[class_id] = next(iter(branch_maps.values()))

        return cams_per_class

    def _compute_score_cam(self, inputs: torch.Tensor, class_ids: List[int]):
        cams_per_class: Dict[int, Dict[str, torch.Tensor] | torch.Tensor] = {}
        original_activations = [act.detach().clone() if act is not None else None for act in self.activations]
        self._validate_single_sample(inputs, "Score-CAM")

        h_in, w_in = inputs.shape[-2:]
        for class_id in class_ids:
            branch_maps = {}
            for branch_idx, activation in enumerate(original_activations):
                if activation is None:
                    continue
                activation = activation.detach()
                self._validate_single_sample(activation, "Score-CAM activations")
                b, c, h, w = activation.shape
                if c == 0:
                    continue
                mean_activation = activation.mean(dim=(2, 3)).squeeze(0)
                topk = min(self.scorecam_topk, c) if self.scorecam_topk is not None else c
                if topk <= 0:
                    continue
                _, top_indices = torch.topk(mean_activation, k=topk)

                cam_branch = torch.zeros((1, 1, h, w), device=activation.device, dtype=activation.dtype)
                valid = False
                for ch_idx in top_indices.tolist():
                    fmap = activation[:, ch_idx:ch_idx + 1, :, :]
                    upsampled = F.interpolate(fmap, size=(h_in, w_in), mode="bilinear", align_corners=False)
                    upsampled = upsampled.squeeze()
                    norm = upsampled - upsampled.min()
                    denom = norm.max()
                    if denom <= 1e-6:
                        continue
                    norm = norm / (denom + 1e-6)
                    mask = norm.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                    with torch.no_grad():
                        self._suppress_hooks = True
                        try:
                            masked_input = inputs * mask
                            masked_output = self.model(masked_input)
                        finally:
                            self._suppress_hooks = False
                    score = masked_output[:, class_id, :, :].mean().item()
                    cam_branch += score * fmap
                    valid = True

                if not valid:
                    continue

                cam_branch = torch.relu(cam_branch)
                cam_branch = F.interpolate(cam_branch, size=(h_in, w_in), mode="bilinear", align_corners=False)
                cam_branch = cam_branch.squeeze()
                cam_branch = (cam_branch - cam_branch.min()) / (cam_branch.max() - cam_branch.min() + 1e-8)

                if self.multi_branch:
                    label = "pos" if branch_idx == 0 else ("neg" if branch_idx == 1 else f"branch_{branch_idx}")
                    branch_maps[label] = cam_branch.cpu()
                else:
                    branch_maps["cam"] = cam_branch.cpu()

            if not branch_maps:
                raise RuntimeError("Score-CAM computation failed: empty CAM map.")

            if self.multi_branch:
                cams_per_class[class_id] = branch_maps
            else:
                cams_per_class[class_id] = next(iter(branch_maps.values()))

        self.activations = tuple(original_activations)
        self.multi_branch = len([act for act in self.activations if act is not None]) > 1
        self.gradients = None
        return cams_per_class

    def generate(
        self,
        inputs: torch.Tensor,
        default_class: Optional[int],
        target_classes: Optional[Sequence[int]] = None,
        include_all: bool = False,
    ) -> tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor] | torch.Tensor]]:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)

        num_channels = outputs.shape[1]
        class_ids: List[int]
        if include_all:
            class_ids = list(range(num_channels))
        elif target_classes is not None and len(target_classes) > 0:
            class_ids = sorted({int(cls) for cls in target_classes})
        elif default_class is not None:
            class_ids = [default_class]
        else:
            preds = outputs.argmax(dim=1)
            classes, counts = preds.unique(return_counts=True)
            if classes.numel() == 0:
                class_ids = [0]
            else:
                order = torch.argsort(counts, descending=True)
                selected = None
                for idx in order:
                    cls_id = classes[idx].item()
                    if cls_id != 0:
                        selected = cls_id
                        break
                target_class = selected if selected is not None else classes[order[0]].item()
                class_ids = [target_class]

        class_ids = [cls_id for cls_id in class_ids if 0 <= cls_id < num_channels]
        if not class_ids:
            raise ValueError("No valid class ids selected for CAM computation.")

        if self.mode == "scorecam":
            cams_per_class = self._compute_score_cam(inputs, class_ids)
        else:
            cams_per_class = self._compute_grad_based_cam(inputs, outputs, class_ids)

        return outputs, cams_per_class


def build_model(model_type: str, n_classes: int, hyper: dict) -> torch.nn.Module:
    depth = int(hyper.get("depth", 4))
    base_pos = int(hyper.get("base_pos", 32))

    if model_type == "unet":
        return build_unet(in_ch=3, n_classes=n_classes, base_ch=base_pos, depth=depth, bilinear=True)
    if model_type == "unet_hybrid":
        base_neg = int(hyper.get("base_neg", 8))
        activation = hyper.get("activation", "tanh")
        if isinstance(activation, str):
            activation = activation.lower()
        orth_val = hyper.get("orth", False)
        orth = bool(orth_val if isinstance(orth_val, (bool, int)) else str(orth_val).lower() in {"1", "true", "yes"})
        proj_mode = hyper.get("proj_mode", "sub")
        dec_base = int(hyper.get("dec_base", 64))
        return build_unet_hybrid_jenc(
            in_ch=3,
            n_classes=n_classes,
            base_pos=base_pos,
            base_neg=base_neg,
            depth=depth,
            act=activation,
            orth=orth,
            proj_mode=proj_mode,
            dec_base=dec_base,
        )
    if model_type == "fcn":
        return build_fcn(in_ch=3, n_classes=n_classes, base_ch=base_pos, stages=depth)
    if model_type == "fcn_hybrid":
        base_neg = int(hyper.get("base_neg", 8))
        activation = hyper.get("activation", "tanh")
        if isinstance(activation, str):
            activation = activation.lower()
        proj_mode = hyper.get("proj_mode", "sub")
        euc_ch = int(hyper.get("euc_ch", 128))
        return build_fcn_hybrid_jenc(
            in_ch=3,
            n_classes=n_classes,
            base_pos=base_pos,
            base_neg=base_neg,
            stages=depth,
            proj_mode=proj_mode,
            euc_ch=euc_ch,
            act=activation,
        )
    raise ValueError(f"Unknown model type '{model_type}'.")


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=image.device).view(-1, 1, 1)
    std = torch.tensor(STD, device=image.device).view(-1, 1, 1)
    return image * std + mean


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """Scale CAM tensor to [0, 1] range with small epsilon for stability."""
    cam_min = cam.min()
    cam_max = cam.max()
    return (cam - cam_min) / (cam_max - cam_min + 1e-8)


def make_explainable_image(image: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    """
    Produce an explainable image by element-wise multiplying the denormalized RGB image
    with a normalized single-channel activation map.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C,H,W); got {tuple(image.shape)}")
    if cam.ndim != 2:
        raise ValueError(f"Expected CAM tensor with shape (H,W); got {tuple(cam.shape)}")

    image_denorm = denormalize(image).clamp(0.0, 1.0)
    cam_norm = normalize_cam(cam).to(image_denorm.device)
    explainable = image_denorm * cam_norm.unsqueeze(0)
    return explainable.clamp(0.0, 1.0)


def apply_cam_to_input(image: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    """Apply a CAM to a normalized image tensor (C,H,W) preserving normalization."""
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C,H,W); got {tuple(image.shape)}")
    if cam.ndim != 2:
        raise ValueError(f"Expected CAM tensor with shape (H,W); got {tuple(cam.shape)}")

    cam_norm = normalize_cam(cam).to(image.device, dtype=image.dtype)
    return image * cam_norm.unsqueeze(0)


def _select_cam_from_entry(entry, preferred_branch: Optional[str] = None) -> Optional[torch.Tensor]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        if preferred_branch and preferred_branch in entry:
            return entry[preferred_branch]
        if "cam" in entry:
            return entry["cam"]
        if "pos" in entry:
            return entry["pos"]
        if "neg" in entry:
            return entry["neg"]
        first_key = sorted(entry.keys())[0]
        return entry[first_key]
    return entry


@torch.no_grad()
def compute_information_loss(
    model: torch.nn.Module,
    image: torch.Tensor,
    class_cam_dict: Dict[int, Dict[str, torch.Tensor] | torch.Tensor],
    num_classes: int,
    original_outputs: Optional[torch.Tensor] = None,
    preferred_branch: Optional[str] = None,
    mask: Optional[torch.Tensor] = None,
) -> tuple[dict, dict]:
    """
    Evaluate the confidence drop per class using CAM-masked inputs specific to each class and
    compute segmentation quality (Dice/IoU) for the explainable predictions.

    Parameters
    ----------
    model: Segmentation model under evaluation.
    image: Normalized input tensor (C,H,W).
    class_cam_dict: Mapping from class id to CAM tensor (single tensor or branch dict).
    num_classes: Number of output classes.
    original_outputs: Optional cached logits from the original image.
    preferred_branch: Optional branch key when CAM entries are multi-branch.
    mask: Optional[torch.Tensor]
        Ground-truth segmentation mask (H,W). Required to compute Dice/IoU for explainable inputs.
    """
    if image.ndim != 3:
        raise ValueError("Image must have shape (C,H,W).")

    model_device = next(model.parameters()).device
    image = image.to(model_device)
    mask_device: Optional[torch.Tensor] = None

    if original_outputs is None:
        outputs_original = model(image.unsqueeze(0))
    else:
        outputs_original = original_outputs.detach()

    prob_original = torch.softmax(outputs_original, dim=1)
    pred_mask = outputs_original.argmax(dim=1).squeeze(0)

    dice_per_class: List[Optional[float]] = [None] * num_classes
    iou_per_class: List[Optional[float]] = [None] * num_classes

    if mask is not None:
        mask_device = mask.to(model_device)

    losses_per_class: List[dict] = []
    weighted_sum = 0.0
    total_weight = 0

    increase_weighted_sum = 0.0
    increase_pixel_sum = 0
    increase_values: List[float] = []

    for cls in range(num_classes):
        class_mask = (pred_mask == cls)
        pixels = int(class_mask.sum().item())
        original_conf = None
        explainable_conf = None
        loss = None
        dice_cls: Optional[float] = None
        iou_cls: Optional[float] = None
        increase_conf: Optional[float] = None
        increase_pixels = 0

        if pixels > 0:
            original_conf = prob_original[0, cls][class_mask].mean().item()

            cam_entry = class_cam_dict.get(cls) if class_cam_dict is not None else None
            cam_tensor = _select_cam_from_entry(cam_entry, preferred_branch=preferred_branch)

            if cam_tensor is not None:
                cam_tensor = cam_tensor.to(model_device)
                masked_input = apply_cam_to_input(image, cam_tensor).unsqueeze(0)
                outputs_explainable = model(masked_input)
                prob_explainable = torch.softmax(outputs_explainable, dim=1)
                explainable_conf = prob_explainable[0, cls][class_mask].mean().item()
                loss = explainable_conf - original_conf
                weighted_sum += loss * pixels
                total_weight += pixels

                if mask_device is not None:
                    dice_vals = dice_score(outputs_explainable, mask_device.unsqueeze(0), n_classes=num_classes, device=model_device)
                    iou_vals = iou_score(outputs_explainable, mask_device.unsqueeze(0), n_classes=num_classes, device=model_device)
                    dice_cls = float(dice_vals[cls].item())
                    iou_cls = float(iou_vals[cls].item())
                    dice_per_class[cls] = dice_cls
                    iou_per_class[cls] = iou_cls

                original_pixels = prob_original[0, cls][class_mask]
                explain_pixels = prob_explainable[0, cls][class_mask]
                deltas = explain_pixels - original_pixels
                positive = deltas[deltas > 0]
                if positive.numel() > 0:
                    increase_conf = float(positive.mean().item())
                    increase_pixels = int(positive.numel())
                    increase_weighted_sum += increase_conf * increase_pixels
                    increase_pixel_sum += increase_pixels
                    increase_values.append(increase_conf)

        losses_per_class.append(
            {
                "class_id": cls,
                "pixels": pixels,
                "original_confidence": original_conf,
                "explainable_confidence": explainable_conf,
                "loss": loss,
                "explainable_dice": dice_cls,
                "explainable_iou": iou_cls,
                "increase_confidence": increase_conf,
                "increase_positive_pixels": increase_pixels,
            }
        )

    mean_loss = None
    if total_weight > 0:
        mean_loss = weighted_sum / total_weight

    valid_losses = [entry["loss"] for entry in losses_per_class if entry["loss"] is not None]
    unweighted_mean = sum(valid_losses) / len(valid_losses) if valid_losses else None

    increase_weighted = None
    if increase_pixel_sum > 0:
        increase_weighted = increase_weighted_sum / increase_pixel_sum
    increase_mean = sum(increase_values) / len(increase_values) if increase_values else None

    dice_values = [val for val in dice_per_class if val is not None]
    iou_values = [val for val in iou_per_class if val is not None]

    explainable_metrics = {
        "dice_per_class": dice_per_class,
        "dice_mean": sum(dice_values) / len(dice_values) if dice_values else None,
        "iou_per_class": iou_per_class,
        "iou_mean": sum(iou_values) / len(iou_values) if iou_values else None,
    }

    return (
        {
            "loss_per_class": losses_per_class,
            "weighted_loss": mean_loss,
            "mean_loss": unweighted_mean,
            "increase_weighted": increase_weighted,
            "increase_mean": increase_mean,
        },
        explainable_metrics,
    )


def compute_segmentation_metrics(
    logits: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> dict:
    """Compute Dice and IoU metrics for a single-sample prediction."""
    if logits.ndim != 4:
        raise ValueError("Expected logits tensor of shape (B,C,H,W).")
    if mask.ndim != 3:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        else:
            raise ValueError("Expected mask tensor of shape (B,H,W) or (H,W).")
    mask = mask.to(device)
    dice = dice_score(logits, mask, n_classes=num_classes, device=device)
    iou = iou_score(logits, mask, n_classes=num_classes, device=device)
    return {
        "dice_per_class": dice.cpu().tolist(),
        "dice_mean": float(dice.mean().cpu()),
        "iou_per_class": iou.cpu().tolist(),
        "iou_mean": float(iou.mean().cpu()),
    }


def save_prediction_image(pred_mask: np.ndarray, output_path: Path, num_classes: int) -> None:
    """Persist a segmentation mask using a spectral colormap."""
    plt.imsave(output_path, pred_mask, cmap="nipy_spectral", vmin=0, vmax=num_classes - 1)


def save_explainable_image(explainable: torch.Tensor, output_path: Path) -> None:
    """Save explainable RGB image (C,H,W) or (H,W,C) as PNG."""
    if explainable.ndim == 3 and explainable.shape[0] in {1, 3}:
        array = explainable.detach().cpu().permute(1, 2, 0).numpy()
    elif explainable.ndim == 3 and explainable.shape[-1] in {1, 3}:
        array = explainable
    else:
        raise ValueError("Explainable image must be (C,H,W) or (H,W,C) with 1 or 3 channels.")
    array = np.clip(array, 0.0, 1.0)
    plt.imsave(output_path, array)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Segmentation Grad-CAM for trained models.")
    parser.add_argument("--model-type", required=True, choices=["unet", "unet_hybrid", "fcn", "fcn_hybrid"], help="Model architecture.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained model weights (.pth).")
    parser.add_argument("--best-params", type=Path, required=True, help="JSON file with best hyperparameters.")
    parser.add_argument("--index", type=int, required=True, help="Dataset index to inspect.")
    parser.add_argument("--target-class", type=int, default=None, help="Class id for Grad-CAM. Defaults to dominant prediction.")
    parser.add_argument(
        "--class-ids",
        nargs="+",
        type=str,
        default=None,
        help="List of class ids to visualize (e.g., 0 1 2). Use 'all' to include every class.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Dataset root directory.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "trainval", "test"], help="Dataset split.")
    parser.add_argument("--img-size", type=int, default=None, help="Resize shorter side to this size. Defaults to training size (256).")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of output classes.")
    parser.add_argument("--output-dir", type=Path, default=Path("./segcam_outputs"), help="Directory to store visualizations.")
    parser.add_argument(
        "--cam-type",
        type=str,
        default="gradcam",
        choices=["gradcam", "gradcam++", "scorecam"],
        help="Type of CAM to compute.",
    )
    parser.add_argument(
        "--scorecam-topk",
        type=int,
        default=32,
        help="Number of activation maps to sample per branch when using Score-CAM.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--base-pos", type=int, default=None, help="Override for base_pos when rebuilding the model.")
    parser.add_argument("--base-neg", type=int, default=None, help="Override for base_neg when rebuilding the model.")
    parser.add_argument("--depth", type=int, default=None, help="Override for model depth.")
    parser.add_argument("--activation", type=str, default=None, help="Override for activation used in hybrid models.")
    parser.add_argument("--proj-mode", type=str, default=None, choices=["sub", "concat"], help="Override projection mode for hybrid models.")
    parser.add_argument("--dec-base", type=int, default=None, help="Override decoder base width for hybrid UNet.")
    parser.add_argument("--orth", dest="orth", action="store_true", help="Force orthogonal J-Conv layers when rebuilding.")
    parser.add_argument("--no-orth", dest="orth", action="store_false", help="Disable orthogonal J-Conv layers when rebuilding.")
    parser.add_argument(
        "--target-layer-name",
        type=str,
        default=None,
        help="Dotted path to the module used for Grad-CAM (e.g., 'dec_convs.0'). Defaults to a sensible last decoder block.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Compute Grad-CAM maps for every output class (overrides --target-class when set).",
    )
    parser.set_defaults(orth=None)
    return parser.parse_args()


def infer_missing_hyperparameters(
    state_dict: dict,
    hyper: dict,
    locked: set[str],
    param_sources: Dict[str, str],
    model_type: str,
):
    if model_type != "unet_hybrid":
        return

    if "depth" not in locked:
        encoder_blocks = {
            key.split(".")[1]
            for key in state_dict.keys()
            if key.startswith("encoder.")
        }
        if encoder_blocks:
            try:
                derived_depth = 1 + max(int(idx) for idx in encoder_blocks)
                if "depth" not in hyper:
                    hyper["depth"] = derived_depth
                    param_sources["depth"] = "checkpoint"
                elif hyper["depth"] != derived_depth:
                    print(f"[WARN] depth mismatch: JSON={hyper['depth']} vs checkpoint={derived_depth}. Using checkpoint value.")
                    hyper["depth"] = derived_depth
                    param_sources["depth"] = "checkpoint (override)"
            except ValueError:
                pass

    if "base_pos" not in locked:
        lift_pos = state_dict.get("lift.lift_pos.weight")
        if lift_pos is not None:
            derived = lift_pos.shape[0]
            if "base_pos" not in hyper:
                hyper["base_pos"] = derived
                param_sources["base_pos"] = "checkpoint"
            elif hyper["base_pos"] != derived:
                print(f"[WARN] base_pos mismatch: JSON={hyper['base_pos']} vs checkpoint={derived}. Using checkpoint value.")
                hyper["base_pos"] = derived
                param_sources["base_pos"] = "checkpoint (override)"
    if "base_neg" not in locked:
        lift_neg = state_dict.get("lift.lift_neg.weight")
        if lift_neg is not None:
            derived = lift_neg.shape[0]
            if "base_neg" not in hyper:
                hyper["base_neg"] = derived
                param_sources["base_neg"] = "checkpoint"
            elif hyper["base_neg"] != derived:
                print(f"[WARN] base_neg mismatch: JSON={hyper['base_neg']} vs checkpoint={derived}. Using checkpoint value.")
                hyper["base_neg"] = derived
                param_sources["base_neg"] = "checkpoint (override)"
    if "orth" not in locked:
        orth_flag = any("R_param" in key for key in state_dict.keys())
        if "orth" not in hyper:
            hyper["orth"] = orth_flag
            param_sources["orth"] = "checkpoint"
        elif bool(hyper["orth"]) != bool(orth_flag):
            print(f"[WARN] orth flag mismatch: JSON={hyper['orth']} vs checkpoint={orth_flag}. Using checkpoint value.")
            hyper["orth"] = bool(orth_flag)
            param_sources["orth"] = "checkpoint (override)"
    if "proj_mode" not in locked:
        if any(key.startswith("proj.head_pos") for key in state_dict.keys()):
            if "proj_mode" not in hyper:
                hyper["proj_mode"] = "sub"
                param_sources["proj_mode"] = "checkpoint"
    if "dec_base" not in locked:
        head_weight = state_dict.get("proj.head_pos.weight")
        if head_weight is not None:
            derived = head_weight.shape[0]
            if "dec_base" not in hyper:
                hyper["dec_base"] = derived
                param_sources["dec_base"] = "checkpoint"
            elif hyper["dec_base"] != derived:
                print(f"[WARN] dec_base mismatch: JSON={hyper['dec_base']} vs checkpoint={derived}. Using checkpoint value.")
                hyper["dec_base"] = derived
                param_sources["dec_base"] = "checkpoint (override)"


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyper = load_hyperparameters(args.best_params)
    param_sources: Dict[str, str] = {key: "json" for key in hyper.keys()}
    locked_keys = set()
    if args.base_pos is not None:
        hyper["base_pos"] = args.base_pos
        locked_keys.add("base_pos")
        param_sources["base_pos"] = "cli"
    if args.base_neg is not None:
        hyper["base_neg"] = args.base_neg
        locked_keys.add("base_neg")
        param_sources["base_neg"] = "cli"
    if args.depth is not None:
        hyper["depth"] = args.depth
        locked_keys.add("depth")
        param_sources["depth"] = "cli"
    if args.activation is not None:
        hyper["activation"] = args.activation
        locked_keys.add("activation")
        param_sources["activation"] = "cli"
    if args.proj_mode is not None:
        hyper["proj_mode"] = args.proj_mode
        locked_keys.add("proj_mode")
        param_sources["proj_mode"] = "cli"
    if args.dec_base is not None:
        hyper["dec_base"] = args.dec_base
        locked_keys.add("dec_base")
        param_sources["dec_base"] = "cli"
    if args.orth is not None:
        hyper["orth"] = args.orth
        locked_keys.add("orth")
        param_sources["orth"] = "cli"
    if args.img_size is None:
        args.img_size = int(hyper.get("img_size", 256))

    state_dict = torch.load(args.weights, map_location="cpu")
    infer_missing_hyperparameters(state_dict, hyper, locked_keys, param_sources, args.model_type)

    tracked_keys = ["base_pos", "base_neg", "depth", "activation", "proj_mode", "dec_base", "orth"]
    resolved = []
    for key in tracked_keys:
        if key in hyper:
            source = param_sources.get(key, "default")
            resolved.append(f"{key}={hyper[key]} ({source})")
    if resolved:
        print(f"Rebuilding {args.model_type} with hyperparameters: {', '.join(resolved)}")

    model = build_model(args.model_type, args.num_classes, hyper)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transforms = PetDatasetTransforms(size=args.img_size)
    dataset = PetDatasetWrapper(root=str(args.data_root), split=args.split, transform=transforms, download=True)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index {args.index} is out of range for split '{args.split}' with {len(dataset)} samples.")

    image, mask = dataset[args.index]
    image_batch = image.unsqueeze(0).to(device)

    cam_type_label_map = {
        "gradcam": "Grad-CAM",
        "gradcam++": "Grad-CAM++",
        "scorecam": "Score-CAM",
    }
    cam_type_label = cam_type_label_map.get(args.cam_type, args.cam_type)
    print(f"Selected CAM type: {cam_type_label}")

    target_layer = resolve_target_layer(model, args.model_type, override=args.target_layer_name)
    if args.target_layer_name:
        print(f"Using custom target layer '{args.target_layer_name}' for {cam_type_label}.")
    else:
        print(f"Using default target layer '{target_layer.__class__.__name__}' for {cam_type_label}.")

    class_id_list: Optional[List[int]] = None
    include_all = args.all_classes
    if args.class_ids:
        tokens = [token.lower() for token in args.class_ids]
        if len(tokens) == 1 and tokens[0] == "all":
            include_all = True
        else:
            try:
                class_id_list = sorted({int(token) for token in args.class_ids})
            except ValueError as exc:
                raise ValueError("--class-ids must be integers or 'all'.") from exc

    cam_helper = SegmentationCAM(model, target_layer, mode=args.cam_type, scorecam_topk=args.scorecam_topk)
    with torch.enable_grad():
        outputs, cam_dict = cam_helper.generate(
            image_batch,
            default_class=args.target_class,
            target_classes=class_id_list,
            include_all=include_all,
        )
    cam_helper.remove()

    if not cam_dict:
        raise RuntimeError("No CAM maps were generated.")

    preds = outputs.argmax(dim=1).squeeze().cpu()
    image_np = denormalize(image).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    mask_np = mask.cpu().numpy()
    pred_np = preds.numpy()

    class_keys = sorted(cam_dict.keys())
    print(f"Computed {cam_type_label} maps for classes: {class_keys}")

    cam_entries = []
    for cls_id in class_keys:
        class_maps = cam_dict[cls_id]
        if isinstance(class_maps, dict):
            for label, cam_tensor in class_maps.items():
                cam_entries.append((f"{cam_type_label} Class {cls_id} {label}", cam_tensor))
        else:
            cam_entries.append((f"{cam_type_label} Class {cls_id}", class_maps))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cam_type_slug = args.cam_type.replace("+", "plus")
    output_path = args.output_dir / f"segcam_{args.model_type}_{cam_type_slug}_idx{args.index}.png"

    total_cols = 3 + len(cam_entries)
    fig, axes = plt.subplots(1, total_cols, figsize=(4 * total_cols, 4))
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[1].imshow(mask_np, cmap="nipy_spectral", vmin=0, vmax=args.num_classes - 1)
    axes[1].set_title("Mask")
    axes[2].imshow(pred_np, cmap="nipy_spectral", vmin=0, vmax=args.num_classes - 1)
    axes[2].set_title("Prediction")

    for plot_idx, (label, cam_tensor) in enumerate(cam_entries, start=3):
        cam_np = cam_tensor.numpy()
        axes[plot_idx].imshow(image_np)
        axes[plot_idx].imshow(cam_np, cmap="jet", alpha=0.45)
        axes[plot_idx].set_title(label)

    fig.suptitle(f"{cam_type_label} maps for {args.model_type} (sample {args.index})", fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved Seg-CAM visualization to: {output_path}")


if __name__ == "__main__":
    main()
