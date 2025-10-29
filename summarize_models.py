import argparse
from pathlib import Path
import torch
import sys
import pandas as pd

# Asegurarse de que los módulos locales (como compute_segcam) se puedan importar
sys.path.append(str(Path(__file__).parent.resolve()))

from compute_segcam import build_model, load_hyperparameters, infer_missing_hyperparameters


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Cuenta los parámetros entrenables y no entrenables de un modelo.
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
        description="Crea un CSV con un resumen de los modelos, sus hiperparámetros y el número de parámetros.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directorio donde se guardan los modelos y los hiperparámetros.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("./results/models_summary.csv"),
        help="Ruta del archivo CSV de salida con el resumen.",
    )
    parser.add_argument("--num-classes", type=int, default=3, help="Número de clases de salida del modelo (para reconstrucción).")
    args = parser.parse_args()

    if not args.models_dir.exists():
        print(f"Error: El directorio de modelos '{args.models_dir}' no existe.")
        return

    weight_files = sorted(list(args.models_dir.glob("best_model_*.pth")))

    if not weight_files:
        print(f"No se encontraron modelos con el patrón 'best_model_*.pth' en '{args.models_dir}'.")
        return

    print(f"Analizando {len(weight_files)} modelos encontrados en '{args.models_dir}'...")

    model_summaries = []

    for weights_path in weight_files:
        model_name_full = weights_path.stem.replace("best_model_", "")
        params_path = args.models_dir / f"best_hyperparameter_{model_name_full}.json"

        if not params_path.exists():
            print(f"AVISO: No se encontró el archivo de hiperparámetros para '{model_name_full}'. Saltando modelo.")
            continue

        # 1. Cargar hiperparámetros desde JSON
        hyper = load_hyperparameters(params_path)

        # 2. Reconstruir el modelo para contar parámetros
        model_type = model_name_full.replace("_orth", "")
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Usamos una copia de los hiperparámetros para no modificarlos
        build_hyper = hyper.copy()
        infer_missing_hyperparameters(state_dict, build_hyper, set(), {}, model_type)
        
        model = build_model(model_type, args.num_classes, build_hyper)
        model.load_state_dict(state_dict)
        model.eval()

        # 3. Contar parámetros
        trainable, non_trainable = count_parameters(model)
        total_params = trainable + non_trainable

        # 4. Preparar la fila de datos para este modelo
        model_data = {
            "model_name": model_name_full,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "total_params": total_params,
        }
        # Añadir los hiperparámetros al diccionario
        model_data.update(hyper)
        
        model_summaries.append(model_data)

    if not model_summaries:
        print("No se pudo procesar ningún modelo.")
        return

    # Crear un DataFrame de pandas. Pandas manejará automáticamente las columnas faltantes (NaN/None).
    summary_df = pd.DataFrame(model_summaries)

    # Asegurar que el directorio de salida exista
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    print(f"\nResumen de modelos guardado exitosamente en: {args.output_csv}")

if __name__ == "__main__":
    main()