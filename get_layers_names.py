import argparse
from pathlib import Path
import torch

from compute_segcam import build_model, load_hyperparameters, infer_missing_hyperparameters


def main():
    parser = argparse.ArgumentParser(
        description="List all named modules and their dotted paths for a given model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["unet", "unet_hybrid", "fcn", "fcn_hybrid", "jvgg19", "jvgg21"],
        help="Model architecture.",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained model weights (.pth).")
    parser.add_argument("--best-params", type=Path, required=True, help="JSON file with best hyperparameters.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of model output classes.")
    args = parser.parse_args()

    print(f"Loading model '{args.model_type}' from weights: {args.weights}")

    # Reconstruye el modelo con los hiperparámetros correctos del checkpoint
    hyper = load_hyperparameters(args.best_params)
    state_dict = torch.load(args.weights, map_location="cpu")
    infer_missing_hyperparameters(state_dict, hyper, set(), {}, args.model_type)
    model = build_model(args.model_type, args.num_classes, hyper)
    model.load_state_dict(state_dict)

    print("\n--- Registered Module Paths ---")
    # Imprime cada submódulo registrado y su ruta punteada
    for name, _ in model.named_modules():
        print(name)
    print("-----------------------------\n")


if __name__ == "__main__":
    main()
