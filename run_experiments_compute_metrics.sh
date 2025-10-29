#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# ==============================================================================
# Script para ejecutar múltiples experimentos de Seg-CAM en lote.
#
# Uso:
#   1. Asegúrate de que este script tenga permisos de ejecución:
#      chmod +x run_experiments.sh
#   2. Ejecútalo desde tu terminal:
#      ./run_experiments.sh
#
# Puedes personalizar los experimentos en la sección 'EXPERIMENT MATRIX'.
# ==============================================================================

# --- Directorios de configuración ---
OUTPUT_BASE_DIR="segcam_outputs"
RESULTS_BASE_DIR="results"

# --- Función para ejecutar un único experimento ---
run_experiment() {
    # Parámetros de la función
    local model_type="$1"
    local cam_type="$2"
    local target_layer="$3" # Puede estar vacío
    local use_orth="$4"     # 'true' para activar el modo ortogonal

    local model_suffix=""
    local orth_label="no"
    if [ "$use_orth" = "true" ]; then
        model_suffix="_orth"
        orth_label="yes"
    fi

    echo "========================================================================"
    echo "Running Experiment:"
    echo "  - Model:         $model_type"
    echo "  - CAM Type:      $cam_type"
    echo "  - Target Layer:  ${target_layer:-'default'}"
    echo "  - Orthogonal:    $orth_label"
    echo "========================================================================"

    # --- Construir rutas y argumentos ---
    local weights_file="models/best_model_${model_type}${model_suffix}.pth"
    local params_file="models/best_hyperparameter_${model_type}${model_suffix}.json"
    
    local layer_slug=${target_layer//./_} # Reemplaza '.' por '_' para el nombre de archivo
    local output_filename="metrics_${model_type}${model_suffix}_${cam_type}_layer-${layer_slug:-default}.json"
    local output_json_path="${OUTPUT_BASE_DIR}/${output_filename}"

    # Construir el comando base
    local cmd=(
        python compute_segcam_metrics_batch.py
        --model-type "$model_type"
        --weights "$weights_file"
        --best-params "$params_file"
        --split "test"
        --all-classes
        --cam-type "$cam_type"
        --output-json "$output_json_path"
        #--save-images # Opcional: guarda imágenes. Quítalo si no lo necesitas.
    )

    # Añadir el target-layer-name solo si se ha especificado
    if [ -n "$target_layer" ]; then
        cmd+=(--target-layer-name "$target_layer")
    fi

    if [ "$use_orth" = "true" ]; then
        cmd+=(--orth)
    fi

    # --- Ejecutar el comando ---
    echo "Executing command:"
    # Imprime el comando antes de ejecutarlo para depuración
    echo "${cmd[@]}"
    "${cmd[@]}"

    # --- Tabular los resultados en un CSV ---
    local csv_output_path="${RESULTS_BASE_DIR}/${output_filename/.json/.csv}"
    echo "Tabulating results to CSV: $csv_output_path"
    python tabulate_metrics.py --input-json "$output_json_path" --output-csv "$csv_output_path"

    echo "Experiment finished successfully."
    echo "========================================================================"
    echo ""
}

# ==============================================================================
# --- MATRIZ DE EXPERIMENTOS ---
#
# Aquí es donde defines todas las ejecuciones que quieres realizar.
# Formato: run_experiment "model_type" "cam_type" "target_layer" "use_orth"
# Deja "target_layer" como "" (comillas vacías) para usar la capa por defecto.
# Pon "true" en el último parámetro para usar la versión ortogonal del modelo.
# ==============================================================================

echo "Starting batch of Seg-CAM experiments..."

# --- Experimentos para unet_hybrid ---
run_experiment "unet_hybrid" "gradcam"   "" "false"
run_experiment "unet_hybrid" "gradcam++" "" "false"
run_experiment "unet_hybrid" "scorecam"  "" "false"
run_experiment "unet_hybrid" "gradcam"   "" "true" # Ejecución con el modelo ortogonal
run_experiment "unet_hybrid" "gradcam++"   "" "true" # Ejecución con el modelo ortogonal
run_experiment "unet_hybrid" "scorecam"   "" "true" # Ejecución con el modelo ortogonal

run_experiment "unet" "gradcam"   "" "false"
run_experiment "unet" "gradcam++" "" "false"
run_experiment "unet" "scorecam"  "" "false"

# run_experiment "fcn_hybrid" "gradcam"   "" "false"
# run_experiment "fcn_hybrid" "gradcam++" "" "false"
# run_experiment "fcn_hybrid" "scorecam"  "" "false"

# run_experiment "fcn_hybrid" "gradcam"   "" "true"
# run_experiment "fcn_hybrid" "gradcam++" "" "true"
# run_experiment "fcn_hybrid" "scorecam"  "" "true"

# run_experiment "fcn" "gradcam"   "" "false"
# run_experiment "fcn" "gradcam++" "" "false"
# run_experiment "fcn" "scorecam"  "" "false"


# --- Experimentos para otros modelos (descomenta para ejecutarlos) ---
# run_experiment "unet" "gradcam" "" "false"
# run_experiment "fcn"  "gradcam" "" "false"

echo "All experiments completed!"
