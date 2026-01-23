import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from pipeline.preprocess import filter_signal, segment_signal, create_reference_signal
from pipeline.alignment import align_segments
from pipeline.feature_extraction import extract_vectorized_features
from pipeline.rf_model import train_rf_model, predict_rf_model, get_feature_importance


# Default pipeline parameters
DEFAULT_PARAMS = {
    'fs': 250,
    'lowcut': 1.0,
    'highcut': 80.0,
    'filter_order': 2,
    'dp_distance': 83,  # fs/3
    'dp_prominence': 0.018,
    'dp_tolerance': 1.5,
    'segment_width': 150,
    'averaging_window': 10,
    'score_percentile_cutoff': 75.0
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
            # Filter
            print("  → Filtering...")
            filtered_df, _ = filter_signal(
                filepath=filepath,
                fs=params['fs'],
                lowcut=params['lowcut'],
                highcut=params['highcut'],
                order=params['filter_order'],
                save=False
            )
            
            # Segment
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
                save=True,
                output_dir=output_dir
            )
            
            if len(segments) == 0:
                print(f"  ✗ No valid segments found, skipping...")
                continue
            
            # Create reference
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
                      Example: {'center': ['center1.csv', 'center2.csv'],
                               'left': ['left1.csv']}
        reference_paths: List of paths to reference CSV files
        params: Dict of pipeline parameters. If None, uses DEFAULT_PARAMS
        
    Returns:
        Dict mapping class names to lists of aligned segment .npy paths
        (Ready to pass to train_rf_model)
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
                # Filter
                print("    → Filtering...")
                filtered_df, _ = filter_signal(
                    filepath=filepath,
                    fs=params['fs'],
                    lowcut=params['lowcut'],
                    highcut=params['highcut'],
                    order=params['filter_order'],
                    save=False
                )
                
                # Segment
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
                    save=True
                )
                
                if len(segments) == 0 or segments_path is None:
                    print(f"    ✗ No valid segments, skipping...")
                    continue
                
                # Align
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
    test_data: Optional[Dict[str, List[str]]] = None,
    params: Optional[Dict] = None,
    model_name: str = "rf_model",
    **model_kwargs
) -> str | None:
    """
    Train a Random Forest model on prepared training data.
    
    Args:
        training_data: Dict mapping class names to aligned .npy file paths
        test_data: Optional dict with same structure for test set
        params: Dict of pipeline parameters for feature extraction
        model_name: Name for saved model file (without .pkl extension)
        **model_kwargs: Additional arguments for train_rf_model
                       (n_estimators, max_depth, etc.)
        
    Returns:
        Path to saved model file
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    print("=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        f"{model_name}.pkl"
    )
    
    model, saved_path = train_rf_model(
        training_data=training_data,
        test_data=test_data,
        score_percentile_cutoff=params['score_percentile_cutoff'],
        model_path=model_path,
        save_model=True,
        **model_kwargs
    )
    
    print(f"\n{'=' * 70}")
    print(f"Model saved: {saved_path}")
    print(f"{'=' * 70}\n")
    
    return saved_path


def predict(
    raw_filepath: str,
    model_path: str,
    reference_paths: List[str],
    params: Optional[Dict] = None,
    return_probabilities: bool = False
) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Make predictions on a new raw data file.
    
    Pipeline: Raw CSV → Filter → Segment → Align → Extract features → Predict
    
    Args:
        raw_filepath: Path to raw CSV file
        model_path: Path to trained model (.pkl)
        reference_paths: List of reference CSV paths
        params: Dict of pipeline parameters
        return_probabilities: If True, return class probabilities
        
    Returns:
        Tuple of (most_common_class, all_predictions, probabilities)
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    print("=" * 70)
    print(f"PREDICTING: {os.path.basename(raw_filepath)}")
    print("=" * 70)
    
    try:
        # Filter
        print("\n→ Filtering...")
        filtered_df, _ = filter_signal(
            filepath=raw_filepath,
            fs=params['fs'],
            lowcut=params['lowcut'],
            highcut=params['highcut'],
            order=params['filter_order'],
            save=False
        )
        
        # Segment
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
            save=True
        )
        
        if len(segments) == 0 or segments_path is None:
            print("✗ No valid segments found")
            return (None, None, None)
        
        print(f"  Found {len(segments)} segments")
        
        # Align
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
        
        # Predict
        print("→ Predicting...")
        most_common, all_preds, probs = predict_rf_model(
            model_path=model_path,
            aligned_data=aligned_results,
            score_percentile_cutoff=params['score_percentile_cutoff'],
            return_probabilities=return_probabilities
        )
        
        print(f"\n{'=' * 70}\n")
        
        return (most_common, all_preds, probs)
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        print(f"{'=' * 70}\n")
        return (None, None, None)


def evaluate_model(
    model_path: str,
    top_n: int = 20
):
    """
    Display model evaluation metrics and feature importance.
    
    Args:
        model_path: Path to trained model
        top_n: Number of top features to display
        
    Returns:
        DataFrame with feature importance
    """
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    print(f"\nModel: {os.path.basename(model_path)}")
    
    print(f"\nTop {top_n} Most Important Features:")
    importance_df = get_feature_importance(model_path, top_n=top_n)
    
    for i in range(len(importance_df)):
        row = importance_df.iloc[i]
        print(f"  {i+1:2d}. {row['feature']:20s} : {row['importance']:.4f}")
    
    print(f"\n{'=' * 70}\n")
    
    return importance_df

if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    MODE = "full"  # Options: "full", "train", "predict"
    
    # File paths
    RAW_TRAINING_FILES = {
        'active': [f"data/raw/excited_{i}.csv" for i in range(1, 6)],
        'relaxed': [f"data/raw/relaxed_{i}.csv" for i in range(1, 7)]
    }
    
    REFERENCE_PATHS = [
        f"data/raw/ref{i}.csv" for i in range(1, 2)
    ]
    
    MODEL_PATH = "models/sensor_classifier_v1.pkl"
    MODEL_NAME = "sensor_classifier_v1"
    
    TEST_FILES = [
        "data/raw/relaxed_7.csv"
    ]
    
    # ========== PIPELINE EXECUTION ==========
    
    if MODE == "full":
        print("\nFULL pipeline (create references + train + predict)\n")
        
        # Create references from training files
        # raw_files = [f for files in RAW_TRAINING_FILES.values() for f in files]
        references = create_references(REFERENCE_PATHS)

        # Prepare training data
        aligned_data = prepare_training_data(RAW_TRAINING_FILES, references)
        
        # Train model
        model_path = train_model(aligned_data, model_name=MODEL_NAME)
        
        # Evaluate
        if model_path is not None:
            evaluate_model(model_path)
            
            # Predict on test files
            for test_file in TEST_FILES:
                most_common, preds, probs = predict(test_file, model_path, references)
    
    elif MODE == "train":
        print("\nTRAINING (using existing references)\n")
        
        # Use existing references
        references = REFERENCE_PATHS
        
        # Prepare training data
        aligned_data = prepare_training_data(RAW_TRAINING_FILES, references)
        
        # Train model
        model_path = train_model(aligned_data, model_name=MODEL_NAME)
        
        # Evaluate
        if model_path is not None:
            evaluate_model(model_path)
    
    elif MODE == "predict":
        print("\n PREDICTION (using existing model and references)\n")
        
        # Use existing references and model
        references = REFERENCE_PATHS
        model_path = MODEL_PATH
        
        # Predict on test files
        for test_file in TEST_FILES:
            most_common, preds, probs = predict(test_file, model_path, references)
    
    else:
        print(f"Unknown: {MODE}")
        print("   Valid options: 'full', 'train', 'predict'")
        