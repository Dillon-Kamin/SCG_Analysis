import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def extract_vectorized_features(
    aligned_results: Optional[List[Tuple]] = None,
    aligned_path: Optional[str] = None,
    score_percentile_cutoff: float = 75.0,
    save: bool = False,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Extract vectorized features from aligned segments.
    
    For each segment:
    - Records Feature 0 amplitude (reference point)
    - Calculates vectors from Feature 0 to all other features
    - Each vector represented as (amplitude, angle, x, y)
    
    Only segments with alignment scores below the percentile cutoff are included.
    Segments without Feature 0 are skipped (cannot calculate vectors).
    
    Args:
        aligned_results: Output from align_segments() as list of tuples:
                        (time, amplitude, features, score, ref_idx)
        aligned_path: Path to .npy file containing aligned results. Alternative to aligned_results.
        score_percentile_cutoff: Only include segments with scores below this percentile.
                                Lower scores are better. Default 75 = best 75%.
        save: If True, save features to CSV.
        output_dir: Directory for saving. If None, uses ../data/features
        filename: Name for saved file. If None, uses 'features.csv'
        
    Returns:
        DataFrame with columns:
        - 'Feature_0_amp': Amplitude of center reference point
        - 'Vector_X_amp': Polar amplitude for feature X
        - 'Vector_X_angle': Polar angle for feature X (radians)
        
        Where X ranges from 1-12 (Feature 0 is the reference point).

        If saved, also returns path to saved CSV file.
    """
    # Load aligned results
    if aligned_results is not None:
        results = aligned_results
    elif aligned_path is not None:
        if not os.path.exists(aligned_path):
            raise FileNotFoundError(f"Aligned results file not found: {aligned_path}")
        results = list(np.load(aligned_path, allow_pickle=True))
    else:
        raise ValueError("Either aligned_results or aligned_path must be provided.")
    
    if len(results) == 0:
        raise ValueError("No aligned results provided.")
    
    # Calculate score cutoff
    scores = np.array([score for _, _, _, score, _ in results])
    cutoff = np.percentile(scores, score_percentile_cutoff)
    
    feature_rows = []
    
    for time, amplitude, feature_array, score, ref_idx in results:
        # Filter by score (lower is better)
        if score >= cutoff:
            continue
        
        # Find all feature points
        feature_mask = feature_array >= 0
        if not np.any(feature_mask):
            continue
        
        feature_indices = np.where(feature_mask)[0]
        feature_times = time[feature_indices]
        feature_amps = amplitude[feature_indices]
        feature_labels = feature_array[feature_indices]
        
        # Find reference point (Feature 0)
        ref_mask = feature_labels == 0
        if not np.any(ref_mask):
            continue  # Skip segments without Feature 0
        
        ref_time = feature_times[ref_mask][0]
        ref_amp = feature_amps[ref_mask][0]
        
        # Initialize feature dict with Feature 0 amplitude
        feature_dict = {'Feature_0_amp': ref_amp}
        
        # Calculate vectors from Feature 0 to all other features
        for feat_time, feat_amp, feat_label in zip(feature_times, feature_amps, feature_labels):
            feat_num = int(feat_label)
            
            # Skip Feature 0 itself (already recorded amplitude)
            if feat_num == 0:
                continue
            
            # Calculate polar coordinates
            dx = feat_time - ref_time
            dy = feat_amp - ref_amp
            amplitude_vec = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            
            # Add all components
            feature_dict[f'Vector_{feat_num}_amp'] = amplitude_vec
            feature_dict[f'Vector_{feat_num}_angle'] = angle
        
        feature_rows.append(feature_dict)
    
    if len(feature_rows) == 0:
        raise ValueError("No valid features extracted. Check score cutoff or alignment quality.")
    
    # Create DataFrame
    df = pd.DataFrame(feature_rows)
    
    # Sort columns for readability: Feature_0_amp, then Vector_1, Vector_2, etc.
    base_cols = ['Feature_0_amp']
    vector_cols = []
    
    for i in range(1, 13):  # Features 1-12
        vector_cols.extend([
            f'Vector_{i}_amp',
            f'Vector_{i}_angle',
        ])
    
    # Only include columns that exist in the dataframe
    ordered_cols = [col for col in base_cols + vector_cols if col in df.columns]
    df = df[ordered_cols]
    
    # Save if requested
    output_path = None
    if save:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/features")
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = 'features.csv'
        
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")
    
    return df, output_path


def load_features(filepath: str) -> pd.DataFrame:
    """
    Load extracted features from CSV.
    
    Args:
        filepath: Path to features CSV file
        
    Returns:
        DataFrame with feature columns
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Features file not found: {filepath}")
    
    return pd.read_csv(filepath)
