import argparse
import json
from pathlib import Path

import torch

from dataset import PetDatasetTransforms, PetDatasetWrapper
from compute_segcam import (
    SegmentationCAM,
    build_model,
    compute_segmentation_metrics,
    infer_missing_hyperparameters,
    load_hyperparameters,
    make_explainable_image,
    resolve_target_layer,
    save_explainable_image,
    save_prediction_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Seg-CAM explainability overlays and segmentation metrics for a dataset sample.",
    )
    parser.add_argument("--model-type", required=True, choices=["unet", "unet_hybrid", "fcn"], help="Model architecture.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained model weights (.pth).")
    parser.add_argument("--best-params", type=Path, required=True, help="JSON file with best hyperparameters.")
    parser.add_argument("--index", type=int, required=True, help="Dataset index to evaluate.")
    parser.add_argument("--target-class", type=int, default=None, help="Default class id for CAM selection.")
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
    parser.add_argument("--split", type=str, default="test", choices=["train", "trainval", "test"], help="Dataset split to load.")
    parser.add_argument("--img-size", type=int, default=None, help="Resize shorter side to this size. Defaults to training size (256).")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of model output classes.")
    parser.add_argument("--output-dir", type=Path, default=Path("./segcam_outputs"), help="Directory to store metrics and images.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. 'cuda' or 'cpu'.")

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
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional explicit path to store metrics as JSON. Defaults to output-dir/<generated-name>.json",
    )
    parser.set_defaults(orth=None)
    return parser.parse_args()


def apply_hyper_overrides(args, hyper: dict) -> tuple[dict, set[str], dict[str, str]]:
    param_sources = {key: "json" for key in hyper.keys()}
    locked_keys: set[str] = set()

    def lock(key: str, value, source: str):
        hyper[key] = value
        locked_keys.add(key)
        param_sources[key] = source

    if args.base_pos is not None:
        lock("base_pos", args.base_pos, "cli")
    if args.base_neg is not None:
        lock("base_neg", args.base_neg, "cli")
    if args.depth is not None:
        lock("depth", args.depth, "cli")
    if args.activation is not None:
        lock("activation", args.activation, "cli")
    if args.proj_mode is not None:
        lock("proj_mode", args.proj_mode, "cli")
    if args.dec_base is not None:
        lock("dec_base", args.dec_base, "cli")
    if args.orth is not None:
        lock("orth", args.orth, "cli")

    return hyper, locked_keys, param_sources


def resolve_class_id_list(args) -> tuple[bool, list[int] | None]:
    include_all = args.all_classes
    class_id_list = None
    if args.class_ids:
        tokens = [token.lower() for token in args.class_ids]
        if len(tokens) == 1 and tokens[0] == "all":
            include_all = True
        else:
            try:
                class_id_list = sorted({int(token) for token in args.class_ids})
            except ValueError as exc:
                raise ValueError("--class-ids must be integers or 'all'.") from exc
    return include_all, class_id_list


def select_cam_entry(cam_dict: dict, preferred_class: int | None, preferred_branch: str | None):
    if not cam_dict:
        raise RuntimeError("No CAM maps were generated.")

    available_classes = sorted(cam_dict.keys())
    if preferred_class is not None and preferred_class in cam_dict:
        class_id = preferred_class
    else:
        class_id = available_classes[0]

    class_maps = cam_dict[class_id]
    if isinstance(class_maps, dict):
        if preferred_branch and preferred_branch in class_maps:
            branch_key = preferred_branch
        elif "cam" in class_maps:
            branch_key = "cam"
        elif "pos" in class_maps:
            branch_key = "pos"
        else:
            branch_key = sorted(class_maps.keys())[0]
        cam_tensor = class_maps[branch_key]
    else:
        branch_key = "cam"
        cam_tensor = class_maps

    return class_id, branch_key, cam_tensor


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
    dataset = PetDatasetWrapper(root=str(args.data_root), split=args.split, transform=transforms, download=True)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index {args.index} is out of range for split '{args.split}' with {len(dataset)} samples.")

    image, mask = dataset[args.index]
    image_batch = image.unsqueeze(0).to(device)

    target_layer = resolve_target_layer(model, args.model_type, override=args.target_layer_name)
    cam_helper = SegmentationCAM(model, target_layer, mode=args.cam_type, scorecam_topk=args.scorecam_topk)

    include_all, class_id_list = resolve_class_id_list(args)
    with torch.enable_grad():
        outputs, cam_dict = cam_helper.generate(
            image_batch,
            default_class=args.target_class,
            target_classes=class_id_list,
            include_all=include_all,
        )
    cam_helper.remove()

    preds = outputs.argmax(dim=1).squeeze(0).cpu()
    metrics = compute_segmentation_metrics(outputs, mask.unsqueeze(0), args.num_classes, device)

    selected_class, branch_key, cam_tensor = select_cam_entry(cam_dict, args.target_class, args.cam_branch)
    explainable = make_explainable_image(image, cam_tensor)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_type_slug = args.cam_type.replace("+", "plus")
    base_name = f"segcam_metrics_{args.model_type}_{cam_type_slug}_idx{args.index}_cls{selected_class}_{branch_key}"

    pred_path = output_dir / f"{base_name}_prediction.png"
    explainable_path = output_dir / f"{base_name}_explainable.png"
    metrics_path = args.metrics_json if args.metrics_json else output_dir / f"{base_name}_metrics.json"

    save_prediction_image(preds.numpy(), pred_path, args.num_classes)
    save_explainable_image(explainable, explainable_path)

    metrics_payload = {
        "model_type": args.model_type,
        "cam_type": args.cam_type,
        "dataset_split": args.split,
        "sample_index": args.index,
        "target_class": args.target_class,
        "selected_cam_class": selected_class,
        "cam_branch": branch_key,
        "prediction_path": str(pred_path),
        "explainable_path": str(explainable_path),
        "metrics": metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Seg-CAM metrics:")
    print(json.dumps(metrics_payload, indent=2))
    print(f"Saved prediction mask to: {pred_path}")
    print(f"Saved explainable image to: {explainable_path}")
    print(f"Saved metrics JSON to: {metrics_path}")


if __name__ == "__main__":
    main()
