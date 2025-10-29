import argparse
import json
from pathlib import Path
from datetime import datetime

import torch

from dataset import PetDatasetTransforms, PetDatasetWrapper
from compute_segcam import (
    SegmentationCAM,
    build_model,
    compute_information_loss,
    compute_segmentation_metrics,
    infer_missing_hyperparameters,
    load_hyperparameters,
    make_explainable_image,
    resolve_target_layer,
    save_explainable_image,
    save_prediction_image,
)
from compute_segcam_metrics import (
    apply_hyper_overrides,
    resolve_class_id_list,
    select_cam_entry,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Seg-CAM explainability metrics for an entire dataset split.",
    )
    parser.add_argument("--model-type", required=True, choices=["unet", "unet_hybrid", "fcn", "fcn_hybrid"], help="Model architecture.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained model weights (.pth).")
    parser.add_argument("--best-params", type=Path, required=True, help="JSON file with best hyperparameters.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "trainval", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--target-class", type=int, default=None, help="Default class id for CAM selection when none is provided.")
    parser.add_argument(
        "--class-ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific class ids to evaluate CAMs for (e.g., 0 1 2). Use 'all' to include every class.",
    )
    parser.add_argument("--all-classes", action="store_true", help="Compute CAMs for all output classes.")
    parser.add_argument("--cam-type", type=str, default="gradcam", choices=["gradcam", "gradcam++", "scorecam"], help="Type of CAM to compute.")
    parser.add_argument("--cam-branch", type=str, default=None, help="Optional branch key when multi-branch CAMs are returned (e.g., 'pos').")
    parser.add_argument("--scorecam-topk", type=int, default=32, help="Top-k activations to sample per branch when using Score-CAM.")
    parser.add_argument("--data-root", type=Path, default=Path("./data"), help="Dataset root directory.")
    parser.add_argument("--img-size", type=int, default=None, help="Resize shorter side to this size. Defaults to training size (256).")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of model output classes.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples to evaluate (processed sequentially).")
    parser.add_argument("--output-json", type=Path, required=True, help="Destination JSON file to store aggregated metrics.")
    parser.add_argument("--output-dir", type=Path, default=Path("./segcam_outputs"), help="Directory to store per-sample images when saving is enabled.")
    parser.add_argument("--save-images", action="store_true", help="Persist prediction and explainable images for each sample.")

    # Optional hyperparameter overrides (mirrors compute_segcam).
    parser.add_argument("--base-pos", type=int, default=None, help="Override base_pos when rebuilding the model.")
    parser.add_argument("--base-neg", type=int, default=None, help="Override base_neg when rebuilding the model.")
    parser.add_argument("--depth", type=int, default=None, help="Override model depth.")
    parser.add_argument("--activation", type=str, default=None, help="Override activation for hybrid models.")
    parser.add_argument("--proj-mode", type=str, default=None, choices=["sub", "concat"], help="Override projection mode for hybrid models.")
    parser.add_argument("--dec-base", type=int, default=None, help="Override decoder base width for hybrid UNet.")
    parser.add_argument("--orth", dest="orth", action="store_true", help="Force orthogonal J-Conv layers when rebuilding.")
    parser.add_argument("--no-orth", dest="orth", action="store_false", help="Disable orthogonal J-Conv layers when rebuilding.")
    parser.add_argument(
        "--target-layer-name",
        type=str,
        default=None,
        help="Dotted path to the module used for CAM (e.g., 'dec_convs.0'). Defaults to a sensible decoder block.",
    )

    parser.set_defaults(orth=None)
    return parser.parse_args()


def init_segmentation_stats(num_classes: int) -> dict:
    return {
        "dice_per_class_sum": [0.0] * num_classes,
        "dice_per_class_count": [0] * num_classes,
        "iou_per_class_sum": [0.0] * num_classes,
        "iou_per_class_count": [0] * num_classes,
        "dice_mean_sum": 0.0,
        "dice_mean_count": 0,
        "iou_mean_sum": 0.0,
        "iou_mean_count": 0,
    }


def init_aggregator(num_classes: int) -> dict:
    return {
        "segmentation": init_segmentation_stats(num_classes),
        "explainable_segmentation": init_segmentation_stats(num_classes),
        "information_loss": {
            "per_class": [
                {
                    "weighted_sum": 0.0,
                    "pixel_sum": 0,
                    "loss_sum": 0.0,
                    "count": 0,
                    "increase_sum": 0.0,
                    "increase_count": 0,
                    "increase_weighted_sum": 0.0,
                    "increase_pixel_sum": 0,
                }
                for _ in range(num_classes)
            ],
            "global_weighted_sum": 0.0,
            "global_pixel_sum": 0,
            "global_loss_sum": 0.0,
            "global_count": 0,
            "global_increase_sum": 0.0,
            "global_increase_count": 0,
            "global_increase_weighted_sum": 0.0,
            "global_increase_pixel_sum": 0,
        },
    }


def update_segmentation_stats(stats: dict, metrics: dict):
    if not metrics:
        return

    dice_list = metrics.get("dice_per_class", [])
    for idx, value in enumerate(dice_list):
        if value is None:
            continue
        stats["dice_per_class_sum"][idx] += float(value)
        stats["dice_per_class_count"][idx] += 1

    iou_list = metrics.get("iou_per_class", [])
    for idx, value in enumerate(iou_list):
        if value is None:
            continue
        stats["iou_per_class_sum"][idx] += float(value)
        stats["iou_per_class_count"][idx] += 1

    dice_mean = metrics.get("dice_mean")
    if dice_mean is not None:
        stats["dice_mean_sum"] += float(dice_mean)
        stats["dice_mean_count"] += 1

    iou_mean = metrics.get("iou_mean")
    if iou_mean is not None:
        stats["iou_mean_sum"] += float(iou_mean)
        stats["iou_mean_count"] += 1


def update_aggregator(aggregator: dict, segmentation: dict, info_loss: dict, explainable_segmentation: dict):
    update_segmentation_stats(aggregator["segmentation"], segmentation)
    update_segmentation_stats(aggregator["explainable_segmentation"], explainable_segmentation)

    info_stats = aggregator["information_loss"]
    for entry in info_loss.get("loss_per_class", []):
        cls = int(entry.get("class_id", -1))
        loss = entry.get("loss")
        pixels = entry.get("pixels")
        if loss is None or pixels is None or cls < 0:
            continue
        class_stats = info_stats["per_class"][cls]
        class_stats["weighted_sum"] += float(loss) * int(pixels)
        class_stats["pixel_sum"] += int(pixels)
        class_stats["loss_sum"] += float(loss)
        class_stats["count"] += 1

        info_stats["global_weighted_sum"] += float(loss) * int(pixels)
        info_stats["global_pixel_sum"] += int(pixels)
        info_stats["global_loss_sum"] += float(loss)
        info_stats["global_count"] += 1

        increase = entry.get("increase_confidence")
        increase_pixels = entry.get("increase_positive_pixels", 0) or 0
        if increase is not None:
            class_stats["increase_sum"] += float(increase)
            class_stats["increase_count"] += 1
            class_stats["increase_weighted_sum"] += float(increase) * int(increase_pixels)
            class_stats["increase_pixel_sum"] += int(increase_pixels)
            info_stats["global_increase_sum"] += float(increase)
            info_stats["global_increase_count"] += 1
            info_stats["global_increase_weighted_sum"] += float(increase) * int(increase_pixels)
            info_stats["global_increase_pixel_sum"] += int(increase_pixels)


def compute_summary_from_stats(stats: dict) -> dict:
    return {
        "dice_per_class_mean": [
            (stats["dice_per_class_sum"][idx] / stats["dice_per_class_count"][idx])
            if stats["dice_per_class_count"][idx] > 0
            else None
            for idx in range(len(stats["dice_per_class_sum"]))
        ],
        "iou_per_class_mean": [
            (stats["iou_per_class_sum"][idx] / stats["iou_per_class_count"][idx])
            if stats["iou_per_class_count"][idx] > 0
            else None
            for idx in range(len(stats["iou_per_class_sum"]))
        ],
        "dice_mean": (
            stats["dice_mean_sum"] / stats["dice_mean_count"]
            if stats["dice_mean_count"] > 0
            else None
        ),
        "iou_mean": (
            stats["iou_mean_sum"] / stats["iou_mean_count"]
            if stats["iou_mean_count"] > 0
            else None
        ),
    }


def finalize_summary(aggregator: dict) -> dict:
    seg_stats = aggregator["segmentation"]
    explain_stats = aggregator["explainable_segmentation"]
    info_stats = aggregator["information_loss"]

    segmentation_summary = compute_summary_from_stats(seg_stats)
    explainable_summary = compute_summary_from_stats(explain_stats)

    per_class_info = []
    for cls, stats in enumerate(info_stats["per_class"]):
        weighted_loss = (
            stats["weighted_sum"] / stats["pixel_sum"]
            if stats["pixel_sum"] > 0
            else None
        )
        mean_loss = (
            stats["loss_sum"] / stats["count"]
            if stats["count"] > 0
            else None
        )
        increase_weighted = (
            stats["increase_weighted_sum"] / stats["increase_pixel_sum"]
            if stats["increase_pixel_sum"] > 0
            else None
        )
        increase_mean = (
            stats["increase_sum"] / stats["increase_count"]
            if stats["increase_count"] > 0
            else None
        )
        per_class_info.append(
            {
                "class_id": cls,
                "weighted_loss": weighted_loss,
                "mean_loss": mean_loss,
                "pixels": stats["pixel_sum"],
                "samples": stats["count"],
                "increase_weighted": increase_weighted,
                "increase_mean": increase_mean,
                "increase_pixels": stats["increase_pixel_sum"],
                "increase_samples": stats["increase_count"],
            }
        )

    information_summary = {
        "per_class": per_class_info,
        "global_weighted_loss": (
            info_stats["global_weighted_sum"] / info_stats["global_pixel_sum"]
            if info_stats["global_pixel_sum"] > 0
            else None
        ),
        "global_mean_loss": (
            info_stats["global_loss_sum"] / info_stats["global_count"]
            if info_stats["global_count"] > 0
            else None
        ),
        "total_pixels": info_stats["global_pixel_sum"],
        "total_measurements": info_stats["global_count"],
        "global_increase_weighted": (
            info_stats["global_increase_weighted_sum"] / info_stats["global_increase_pixel_sum"]
            if info_stats["global_increase_pixel_sum"] > 0
            else None
        ),
        "global_increase_mean": (
            info_stats["global_increase_sum"] / info_stats["global_increase_count"]
            if info_stats["global_increase_count"] > 0
            else None
        ),
        "total_increase_pixels": info_stats["global_increase_pixel_sum"],
        "total_increase_measurements": info_stats["global_increase_count"],
    }

    return {
        "segmentation": segmentation_summary,
        "explainable_segmentation": explainable_summary,
        "information_loss": information_summary,
    }


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hyper = load_hyperparameters(args.best_params)
    hyper, locked_keys, param_sources = apply_hyper_overrides(args, hyper)

    if args.img_size is None:
        args.img_size = int(hyper.get("img_size", 256))

    state_dict = torch.load(args.weights, map_location="cpu")
    infer_missing_hyperparameters(state_dict, hyper, locked_keys, param_sources, args.model_type)

    model = build_model(args.model_type, args.num_classes, hyper)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transforms = PetDatasetTransforms(size=args.img_size)
    dataset_split = "trainval" if args.split == "train" else args.split
    dataset = PetDatasetWrapper(root=str(args.data_root), split=dataset_split, transform=transforms, download=True)

    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError(f"Dataset split '{args.split}' is empty.")

    max_samples = args.max_samples if args.max_samples is not None else total_samples
    indices = list(range(min(max_samples, total_samples)))

    include_all, class_id_list = resolve_class_id_list(args)

    target_layer = resolve_target_layer(model, args.model_type, override=args.target_layer_name)
    cam_helper = SegmentationCAM(model, target_layer, mode=args.cam_type, scorecam_topk=args.scorecam_topk)

    if args.save_images:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    aggregator = init_aggregator(args.num_classes)

    cam_type_slug = args.cam_type.replace("+", "plus")

    for idx in indices:
        image, mask = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)

        with torch.enable_grad():
            outputs, cam_dict = cam_helper.generate(
                image_batch,
                default_class=args.target_class,
                target_classes=class_id_list,
                include_all=include_all,
            )

        outputs = outputs.detach()
        preds = outputs.argmax(dim=1).squeeze(0).cpu()

        segmentation_metrics = compute_segmentation_metrics(outputs, mask.unsqueeze(0), args.num_classes, device)
        info_loss, explainable_metrics = compute_information_loss(
            model,
            image.to(device),
            cam_dict,
            args.num_classes,
            original_outputs=outputs,
            preferred_branch=args.cam_branch,
            mask=mask,
        )

        prediction_path = None
        explainable_path = None
        selected_class = None
        branch_key = None

        if args.save_images:
            selected_class, branch_key, cam_tensor = select_cam_entry(cam_dict, args.target_class, args.cam_branch)
            if cam_tensor is not None:
                explainable = make_explainable_image(image, cam_tensor)
                base_name = (
                    f"segcam_metrics_dataset_{args.model_type}_{cam_type_slug}_split{args.split}_idx{idx}_"
                    f"cls{selected_class}_{branch_key}"
                )
                prediction_path = args.output_dir / f"{base_name}_prediction.png"
                explainable_path = args.output_dir / f"{base_name}_explainable.png"
                save_prediction_image(preds.numpy(), prediction_path, args.num_classes)
                save_explainable_image(explainable, explainable_path)

        sample_entry = {
            "index": idx,
            "cam_classes": sorted(int(k) for k in cam_dict.keys()),
            "selected_cam_class": selected_class,
            "cam_branch": branch_key,
            "prediction_path": str(prediction_path) if prediction_path else None,
            "explainable_path": str(explainable_path) if explainable_path else None,
            "metrics": {
                "segmentation": segmentation_metrics,
                "information_loss": info_loss,
                "explainable_segmentation": explainable_metrics,
            },
        }

        update_aggregator(aggregator, segmentation_metrics, info_loss, explainable_metrics)
        results.append(sample_entry)

    cam_helper.remove()

    summary = finalize_summary(aggregator)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_type": args.model_type,
        "cam_type": args.cam_type,
        "dataset_split": args.split,
        "dataset_split_internal": dataset_split,
        "num_samples": len(indices),
        "class_ids": class_id_list,
        "all_classes": include_all,
        "cam_branch": args.cam_branch,
        "save_images": args.save_images,
        "output_dir": str(args.output_dir) if args.save_images else None,
        "samples": results,
        "summary": summary,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Evaluated {len(indices)} samples from split '{args.split}'.")
    print(f"Saved aggregated metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
