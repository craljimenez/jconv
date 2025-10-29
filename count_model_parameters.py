import argparse
from pathlib import Path
import torch
import sys

# Asegurarse de que los módulos locales (como compute_segcam) se puedan importar
# Esto asume que el script se ejecuta desde el directorio raíz del proyecto.
sys.path.append(str(Path(__file__).parent.resolve()))

from compute_segcam import build_model, load_hyperparameters, infer_missing_hyperparameters


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Cuenta los parámetros entrenables y no entrenables de un modelo.

    Args:
        model (torch.nn.Module): El modelo de PyTorch a analizar.

    Returns:
        tuple[int, int]: Una tupla conteniendo (parámetros_entrenables, parámetros_no_entrenables).
    """
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    return trainable_params, non_trainable_params


def main():
    parser = argparse.ArgumentParser(
        description="Carga modelos desde el directorio 'models' y cuenta sus parámetros.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("./models"), help="Directorio donde se guardan los modelos y los hiperparámetros."
    )
    parser.add_argument("--num-classes", type=int, default=3, help="Número de clases de salida del modelo.")
    args = parser.parse_args()

    if not args.models_dir.exists():
        print(f"Error: El directorio de modelos '{args.models_dir}' no existe.")
        return

    # Buscar todos los archivos de pesos de modelos
    weight_files = sorted(list(args.models_dir.glob("best_model_*.pth")))

    if not weight_files:
        print(f"No se encontraron modelos con el patrón 'best_model_*.pth' en '{args.models_dir}'.")
        return

    print(f"Analizando {len(weight_files)} modelos encontrados en '{args.models_dir}'...\n")

    for weights_path in weight_files:
        # Extraer el tipo de modelo y sufijo del nombre del archivo
        model_name_full = weights_path.stem.replace("best_model_", "")
        params_path = args.models_dir / f"best_hyperparameter_{model_name_full}.json"

        if not params_path.exists():
            print(f"--- Modelo: {model_name_full} ---")
            print(f"AVISO: No se encontró el archivo de hiperparámetros '{params_path.name}'. Saltando modelo.\n")
            continue

        # Reconstruir el modelo usando la misma lógica que en otros scripts
        model_type = model_name_full.replace("_orth", "") # Extraer tipo base (e.g., 'unet_hybrid')
        hyper = load_hyperparameters(params_path)
        state_dict = torch.load(weights_path, map_location="cpu")
        infer_missing_hyperparameters(state_dict, hyper, set(), {}, model_type)
        
        model = build_model(model_type, args.num_classes, hyper)
        model.load_state_dict(state_dict)
        model.eval()  # Poner en modo evaluación

        trainable, non_trainable = count_parameters(model)
        total = trainable + non_trainable

        print(f"--- Modelo: {model_name_full} ---")
        print(f"  Parámetros entrenables:   {trainable:,}")
        print(f"  Parámetros no entrenables: {non_trainable:,}")
        print(f"  Total de parámetros:      {total:,}\n")

if __name__ == "__main__":
    main()
