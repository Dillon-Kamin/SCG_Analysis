import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from pipeline.preprocess import filter_signal, segment_signal, create_reference_signal
from pipeline.alignment import align_segments
from pipeline.feature_extraction import extract_vectorized_features
from pipeline.rf_model import train_rf_model, predict_rf_model, get_feature_importance
from pipeline.cnn_model import train_cnn_model, predict_cnn_model


# Default pipeline parameters
DEFAULT_PARAMS = {
    # Signal processing
    'fs': 500,
    'lowcut': 2.0,
    'highcut': 50.0,
    'filter_order': 2,

    # Segmentation
    'dp_distance': 167,           # fs/3
    'dp_prominence': 0.02,
    'dp_tolerance': 1.5,
    'segment_width': 300,
    'averaging_window': 10,

    # AO/MV refinement
    'ao_mv_refinement': True,
    'local_search_radius': 25,
    'local_distance': 5,
    'similarity_threshold': 0.6,

    # Alignment quality (RF only — CNN uses absolute threshold)
    'score_percentile_cutoff': 75.0,

    # CNN-specific
    'crop_start': 100,            # discard uninformative pre-AO samples
    'crop_end': 300,
    'n_epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'dropout_conv': 0.5,
    'dropout_fc': 0.3,
    'val_split': 0.15,
}


def create_references(
    raw_filepaths: List[str],
    params: Optional[Dict] = None,
    output_dir: Optional[str] = None
) -> List[str]:
    """
    Create reference signals from raw data files.

    Pipeline: Raw CSV → Filter → Segment → Average → Label features → Save reference

    Args:
        raw_filepaths: List of paths to raw CSV files
        params: Dict of pipeline parameters. If None, uses DEFAULT_PARAMS
        output_dir: Directory to save references. If None, uses default from create_reference_signal

    Returns:
        List of paths to created reference CSV files
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    reference_paths = []

    print("=" * 70)
    print("CREATING REFERENCE SIGNALS")
    print("=" * 70)

    for filepath in raw_filepaths:
        print(f"\nProcessing: {os.path.basename(filepath)}")

        try:
            print("  → Filtering...")
            filtered_df, _ = filter_signal(
                filepath=filepath,
                fs=params['fs'],
                lowcut=params['lowcut'],
                highcut=params['highcut'],
                order=params['filter_order'],
                save=False
            )

            print("  → Segmenting...")
            segments, segments_path = segment_signal(
                df=filtered_df,
                filepath=filepath,
                fs=params['fs'],
                distance=params['dp_distance'],
                prominence=params['dp_prominence'],
                tolerance=params['dp_tolerance'],
                segment_width=params['segment_width'],
                averaging_window=params['averaging_window'],
                ao_mv_refinement=params['ao_mv_refinement'],
                local_search_radius=params['local_search_radius'],
                local_distance=params['local_distance'],
                similarity_threshold=params['similarity_threshold'],
                save=True,
                output_dir=output_dir
            )

            if len(segments) == 0:
                print(f"  ✗ No valid segments found, skipping...")
                continue

            print("  → Creating reference signal...")
            ref_df, ref_path = create_reference_signal(
                segments=segments,
                filepath=segments_path,
                fs=params['fs'],
                output_dir=output_dir
            )

            if ref_path:
                reference_paths.append(ref_path)
                print(f"  ✓ Reference created: {os.path.basename(ref_path)}")

        except Exception as e:
            print(f"  ✗ Error processing {filepath}: {e}")
            continue

    print(f"\n{'=' * 70}")
    print(f"Created {len(reference_paths)} reference signals")
    print(f"{'=' * 70}\n")

    return reference_paths


def prepare_training_data(
    raw_filepaths: Dict[str, List[str]],
    reference_paths: List[str],
    params: Optional[Dict] = None
) -> Dict[str, List[str]]:
    """
    Prepare training data from raw files.

    Pipeline: Raw CSV → Filter → Segment → Align → Save aligned segments

    Args:
        raw_filepaths: Dict mapping class names to lists of raw CSV paths
        reference_paths: List of paths to reference CSV files
        params: Dict of pipeline parameters. If None, uses DEFAULT_PARAMS

    Returns:
        Dict mapping class names to lists of aligned segment .npy paths
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    aligned_data = {class_name: [] for class_name in raw_filepaths.keys()}

    print("=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)

    for class_name, filepaths in raw_filepaths.items():
        print(f"\nClass: {class_name}")

        for filepath in filepaths:
            print(f"  Processing: {os.path.basename(filepath)}")

            try:
                print("    → Filtering...")
                filtered_df, _ = filter_signal(
                    filepath=filepath,
                    fs=params['fs'],
                    lowcut=params['lowcut'],
                    highcut=params['highcut'],
                    order=params['filter_order'],
                    save=False
                )

                print("    → Segmenting...")
                segments, segments_path = segment_signal(
                    df=filtered_df,
                    filepath=filepath,
                    fs=params['fs'],
                    distance=params['dp_distance'],
                    prominence=params['dp_prominence'],
                    tolerance=params['dp_tolerance'],
                    segment_width=params['segment_width'],
                    averaging_window=params['averaging_window'],
                    ao_mv_refinement=params['ao_mv_refinement'],
                    local_search_radius=params['local_search_radius'],
                    local_distance=params['local_distance'],
                    similarity_threshold=params['similarity_threshold'],
                    save=True
                )

                if len(segments) == 0 or segments_path is None:
                    print(f"    ✗ No valid segments, skipping...")
                    continue

                print("    → Aligning...")
                aligned_results, aligned_path = align_segments(
                    segments_path=segments_path,
                    reference_paths=reference_paths,
                    fs=params['fs'],
                    save=True
                )

                if len(aligned_results) == 0 or aligned_path is None:
                    print(f"    ✗ Alignment failed, skipping...")
                    continue

                aligned_data[class_name].append(aligned_path)
                print(f"    ✓ Aligned {len(aligned_results)} segments")

            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

    print(f"\n{'=' * 70}")
    print("Training data prepared:")
    for class_name, paths in aligned_data.items():
        print(f"  {class_name}: {len(paths)} files")
    print(f"{'=' * 70}\n")

    return aligned_data


def train_model(
    training_data: Dict[str, List[str]],
    reference_aligned_path: str,
    model_type: str = 'cnn',
    test_data: Optional[Dict[str, List[str]]] = None,
    params: Optional[Dict] = None,
    model_name: str = "model",
    **model_kwargs
) -> Optional[str]:
    """
    Train a classifier on prepared training data.

    Args:
        training_data:          Dict mapping class names to aligned .npy paths.
        reference_aligned_path: Path to aligned .npy for the reference recording.
                                 Used by the CNN to calibrate the absolute score
                                 threshold. Not used by RF (which uses per-file
                                 percentile cutoff).
        model_type:             'cnn' or 'rf'.
        test_data:              Optional held-out test set (RF only).
        params:                 Pipeline parameters. If None, uses DEFAULT_PARAMS.
        model_name:             Filename stem for saved model (no extension).
        **model_kwargs:         Additional arguments forwarded to the trainer.

    Returns:
        Path to saved model file, or None on failure.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    print("=" * 70)
    print(f"TRAINING MODEL  [{model_type.upper()}]")
    print("=" * 70)

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models"
    )

    if model_type == 'cnn':
        ext = 'pkl'
        model_path = os.path.join(output_dir, f"{model_name}.{ext}")
        _, saved_path = train_cnn_model(
            training_data=training_data,
            reference_aligned_path=reference_aligned_path,
            score_percentile=params['score_percentile_cutoff'],
            crop_start=params['crop_start'],
            crop_end=params['crop_end'],
            n_epochs=params.get('n_epochs', 50),
            batch_size=params.get('batch_size', 32),
            learning_rate=params.get('learning_rate', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4),
            dropout_conv=params.get('dropout_conv', 0.5),
            dropout_fc=params.get('dropout_fc', 0.3),
            val_split=params.get('val_split', 0.15),
            save_model=True,
            model_path=model_path,
            **model_kwargs
        )

    elif model_type == 'rf':
        ext = 'pkl'
        model_path = os.path.join(output_dir, f"{model_name}.{ext}")
        _, saved_path = train_rf_model(
            training_data=training_data,
            test_data=test_data,
            score_percentile_cutoff=params['score_percentile_cutoff'],
            model_path=model_path,
            save_model=True,
            **model_kwargs
        )

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'cnn' or 'rf'.")

    print(f"\n{'=' * 70}")
    print(f"Model saved: {saved_path}")
    print(f"{'=' * 70}\n")

    return saved_path


def predict(
    raw_filepath: str,
    model_path: str,
    reference_paths: List[str],
    model_type: str = 'cnn',
    params: Optional[Dict] = None,
    return_probabilities: bool = False
) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Make predictions on a new raw data file.

    Pipeline: Raw CSV → Filter → Segment → Align → Predict

    Args:
        raw_filepath:       Path to raw CSV file.
        model_path:         Path to trained model (.pkl).
        reference_paths:    List of reference CSV paths for alignment.
        model_type:         'cnn' or 'rf'.
        params:             Pipeline parameters. If None, uses DEFAULT_PARAMS.
        return_probabilities: If True, return class probabilities (CNN always
                              returns them; RF only if requested).

    Returns:
        Tuple of (most_common_class_idx, all_predictions, probabilities)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    print("=" * 70)
    print(f"PREDICTING [{model_type.upper()}]: {os.path.basename(raw_filepath)}")
    print("=" * 70)

    try:
        print("\n→ Filtering...")
        filtered_df, _ = filter_signal(
            filepath=raw_filepath,
            fs=params['fs'],
            lowcut=params['lowcut'],
            highcut=params['highcut'],
            order=params['filter_order'],
            save=False
        )

        print("→ Segmenting...")
        segments, segments_path = segment_signal(
            df=filtered_df,
            filepath=raw_filepath,
            fs=params['fs'],
            distance=params['dp_distance'],
            prominence=params['dp_prominence'],
            tolerance=params['dp_tolerance'],
            segment_width=params['segment_width'],
            averaging_window=params['averaging_window'],
            ao_mv_refinement=params['ao_mv_refinement'],
            local_search_radius=params['local_search_radius'],
            local_distance=params['local_distance'],
            similarity_threshold=params['similarity_threshold'],
            save=True
        )

        if len(segments) == 0 or segments_path is None:
            print("✗ No valid segments found")
            return (None, None, None)

        print(f"  Found {len(segments)} segments")

        print("→ Aligning...")
        aligned_results, _ = align_segments(
            segments_path=segments_path,
            reference_paths=reference_paths,
            fs=params['fs'],
            save=False
        )

        if len(aligned_results) == 0:
            print("✗ Alignment failed")
            return (None, None, None)

        print(f"  Aligned {len(aligned_results)} segments")

        print("→ Predicting...")
        if model_type == 'cnn':
            most_common, all_preds, probs = predict_cnn_model(
                model_path=model_path,
                aligned_data=aligned_results,
            )
        elif model_type == 'rf':
            most_common, all_preds, probs = predict_rf_model(
                model_path=model_path,
                aligned_data=aligned_results,
                score_percentile_cutoff=params['score_percentile_cutoff'],
                return_probabilities=return_probabilities
            )
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'cnn' or 'rf'.")

        print(f"\n{'=' * 70}\n")
        return (most_common, all_preds, probs)

    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        print(f"{'=' * 70}\n")
        return (None, None, None)


def evaluate_model(
    model_path: str,
    model_type: str = 'rf',
    top_n: int = 20
):
    """
    Display model evaluation metrics and feature importance.
    RF only — CNN does not use hand-crafted features.

    Args:
        model_path: Path to trained model
        model_type: 'rf' or 'cnn'. Feature importance only available for RF.
        top_n:      Number of top features to display (RF only)
    """
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"\nModel: {os.path.basename(model_path)}")

    if model_type == 'rf':
        print(f"\nTop {top_n} Most Important Features:")
        importance_df = get_feature_importance(model_path, top_n=top_n)
        for i in range(len(importance_df)):
            row = importance_df.iloc[i]
            print(f"  {i+1:2d}. {row['feature']:25s} : {row['importance']:.4f}")
        print(f"\n{'=' * 70}\n")
        return importance_df

    elif model_type == 'cnn':
        import joblib
        model_data = joblib.load(model_path)
        print(f"  Classes:         {model_data['class_names']}")
        print(f"  Input width:     {model_data['input_width']} samples  "
              f"[{model_data['crop_start']}:{model_data['crop_end']}]")
        print(f"  Score threshold: {model_data['score_threshold']:.4f}")
        print(f"\n{'=' * 70}\n")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ========== CONFIGURATION ==========
    MODE       = "full"    # "full" | "train" | "predict"
    MODEL_TYPE = "rf"     # "cnn"  | "rf"

    RAW_TRAINING_FILES = {
        'engaged': [f"data/raw/engage{i}.csv" for i in range(2, 5)],
        'relaxed': [f"data/raw/relax{i}.csv"  for i in range(4, 6)],
    }

    # File used to build the reference signal AND calibrate the score threshold
    REFERENCE_RAW = "data/raw/engage5.csv"

    # After running create_references, this will exist:
    REFERENCE_CSV_PATHS = ["data/references/engage5_reference.csv"]

    # Aligned version of the reference file (produced by prepare_training_data
    # or a separate alignment run) — used by CNN for threshold calibration
    REFERENCE_ALIGNED_PATH = "data/aligned/engage5_segments_aligned.npy"

    MODEL_NAME = f"sensor_classifier_v1_{MODEL_TYPE}"
    MODEL_PATH = f"models/{MODEL_NAME}.pkl"

    TEST_FILES = ["data/raw/engage1.csv"]

    # ========== PIPELINE EXECUTION ==========

    if MODE == "full":
        print("\nFULL pipeline (create references + train + predict)\n")

        references = create_references([REFERENCE_RAW])

        aligned_data = prepare_training_data(RAW_TRAINING_FILES, references)

        model_path = train_model(
            training_data=aligned_data,
            reference_aligned_path=REFERENCE_ALIGNED_PATH,
            model_type=MODEL_TYPE,
            model_name=MODEL_NAME,
        )

        if model_path is not None:
            evaluate_model(model_path, model_type=MODEL_TYPE)
            for test_file in TEST_FILES:
                predict(test_file, model_path, references, model_type=MODEL_TYPE)

    elif MODE == "train":
        print("\nTRAINING (using existing references)\n")

        aligned_data = prepare_training_data(RAW_TRAINING_FILES, REFERENCE_CSV_PATHS)

        model_path = train_model(
            training_data=aligned_data,
            reference_aligned_path=REFERENCE_ALIGNED_PATH,
            model_type=MODEL_TYPE,
            model_name=MODEL_NAME,
        )

        if model_path is not None:
            evaluate_model(model_path, model_type=MODEL_TYPE)

    elif MODE == "predict":
        print("\nPREDICTION (using existing model and references)\n")

        for test_file in TEST_FILES:
            predict(test_file, MODEL_PATH, REFERENCE_CSV_PATHS, model_type=MODEL_TYPE)

    else:
        print(f"Unknown mode: {MODE}")
        print("Valid options: 'full', 'train', 'predict'")