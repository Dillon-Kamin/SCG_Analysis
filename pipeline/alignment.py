import os
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple, Optional
import polars as pl


def align_segments(
    segments_path: str,
    reference_paths: List[str],
    fs: float = 500,
    h: float = -0.0125,
    g: float = -0.0025,
    save: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[List[Tuple], Optional[str]]:
    """
    Align segments against reference CSVs using dynamic programming.

    Args:
        segments_path: Path to .npy file containing segments array.
        reference_paths: List of CSV file paths for references.
        fs: Sampling frequency (Hz).
        h: Gap open penalty.
        g: Gap extension penalty.
        save: If True, save alignment results.
        output_dir: Directory to save aligned results. If None, uses ../data/aligned

    Returns:
        List of tuples:
        - (time, amplitude, features, score, ref_idx)
    """
    # Load and preprocess references and segments
    references = _load_references(reference_paths)
    segments = list(np.load(segments_path, allow_pickle=True))

    results = []

    for i, segment in enumerate(segments):
        # Extract peaks from segment
        seg_peaks = _extract_peaks(segment, fs)
        
        # Find best alignment across all references
        best_score = -np.inf
        best_features = None
        best_ref_idx = None
        
        for ref_idx, ref_peaks in enumerate(references):
            features, score = _align_to_reference(ref_peaks, seg_peaks, h, g)
            
            if score > best_score:
                best_score = score
                best_features = features
                best_ref_idx = ref_idx
        
        if best_features is None:
            print(f"Alignment failed for segment {i}.")
            continue

        if best_score == -np.inf:
            print(f"No significant alignment found for segment {i}.")
            continue

        # Map features back to full segment
        segment_features = _map_features_to_segment(
            segment, seg_peaks, best_features, fs
        )
        
        time = np.arange(len(segment)) / fs
        results.append((time, segment, segment_features, -best_score, best_ref_idx))

    # Save if requested (in numpy tuple format)
    output_path = None
    if save:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/aligned")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(segments_path).replace(".npy", "")
        output_path = os.path.join(output_dir, f"{base_name}_aligned.npy")
        
        save_results = results

        np.save(output_path, np.array(results, dtype=object), allow_pickle=True)

    return results, output_path


def _load_references(paths: List[str]) -> List[np.ndarray]:
    """
    Load references and extract peak information as structured arrays.
    
    Returns:
        List of numpy structured arrays with fields:
        - time: float
        - amplitude: float
        - direction: int (0=Down, 1=Up)
        - feature: int
    """
    references = []
    
    for path in paths:
        # Load CSV
        df = pl.read_csv(path)
        
        # Filter to feature points only
        df_features = df.filter(pl.col("Feature") >= 0)
        
        # Determine direction: Up if Feature in [2,4] or (Feature > 4 and odd)
        features = df_features["Feature"].to_numpy()
        directions = np.where(
            np.isin(features, [2, 4]) | ((features > 4) & (features % 2 == 1)),
            1,  # Up
            0   # Down
        )
        
        # Create structured array for fast access
        ref_peaks = np.zeros(
            len(df_features),
            dtype=[('time', 'f8'), ('amplitude', 'f8'), ('direction', 'i4'), ('feature', 'i4')]
        )
        ref_peaks['time'] = df_features["Time"].to_numpy()
        ref_peaks['amplitude'] = df_features["Amplitude"].to_numpy()
        ref_peaks['direction'] = directions
        ref_peaks['feature'] = features
        
        references.append(ref_peaks)
    
    return references


def _extract_peaks(segment: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract peaks from segment.
    
    Returns:
        Structured array with fields: time, amplitude, direction, index
    """
    # Find peaks
    down_idx, _ = find_peaks(-segment)
    up_idx, _ = find_peaks(segment)
    
    # Combine and sort by time
    n_down = len(down_idx)
    n_up = len(up_idx)
    n_total = n_down + n_up
    
    peaks = np.zeros(
        n_total,
        dtype=[('time', 'f8'), ('amplitude', 'f8'), ('direction', 'i4'), ('index', 'i4')]
    )
    
    # Down peaks
    peaks['time'][:n_down] = down_idx / fs
    peaks['amplitude'][:n_down] = segment[down_idx]
    peaks['direction'][:n_down] = 0
    peaks['index'][:n_down] = down_idx
    
    # Up peaks
    peaks['time'][n_down:] = up_idx / fs
    peaks['amplitude'][n_down:] = segment[up_idx]
    peaks['direction'][n_down:] = 1
    peaks['index'][n_down:] = up_idx
    
    # Sort by time
    peaks.sort(order='time')
    
    return peaks


def _align_to_reference(
    ref_peaks: np.ndarray,
    seg_peaks: np.ndarray,
    h: float,
    g: float
) -> Tuple[np.ndarray, float]:
    """
    Align segment peaks to reference peaks using dynamic programming in a modified Needleman-Wunsch algorithm.
    
    Returns:
        (features, score) where features is array of feature IDs for seg_peaks
    """
    m = len(ref_peaks)
    n = len(seg_peaks)
    
    # Allocate DP tables
    S = np.full((m + 1, n + 1), -np.inf, dtype=np.float64)
    I = np.full((m + 1, n + 1), -np.inf, dtype=np.float64)
    
    # Initialize
    S[0, 0] = 0.0
    I[0, 0] = 0.0
    for j in range(1, n + 1):
        I[0, j] = h + (j - 1) * g
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ref = ref_peaks[i - 1]
            seg = seg_peaks[j - 1]
            
            # Match score: -distance if same direction, else -inf
            if ref['direction'] == seg['direction']:
                dx = ref['time'] - seg['time']
                dy = ref['amplitude'] - seg['amplitude']
                dist = np.sqrt(dx * dx + dy * dy)
                match_score = -dist
            else:
                match_score = -np.inf
            
            # Update tables
            S[i, j] = max(S[i-1, j-1], I[i-1, j-1]) + match_score
            I[i, j] = max(S[i, j-1] + h, I[i, j-1] + g)
    
    # Backtrace
    features = np.full(n, -1, dtype=np.int32)
    score = max(S[m, n], I[m, n])
    
    i, j = m, n
    current_table = 0 if S[i, j] >= I[i, j] else 1  # 0=S, 1=I
    
    while i > 0 and j > 0:
        if current_table == 0:  # S table
            features[j - 1] = ref_peaks[i - 1]['feature']
            i -= 1
            j -= 1
            current_table = 0 if S[i, j] >= I[i, j] else 1
        else:  # I table
            j -= 1
            current_table = 0 if S[i, j] + h >= I[i, j] + g else 1
    
    return features, score


def _map_features_to_segment(
    segment: np.ndarray,
    seg_peaks: np.ndarray,
    peak_features: np.ndarray,
    fs: float
) -> np.ndarray:
    """
    Map features from peaks back to full segment array.
    
    Args:
        segment: Full amplitude array
        seg_peaks: Peak information (with 'index' field)
        peak_features: Feature IDs for each peak
        fs: Sampling frequency
    
    Returns:
        Feature array same length as segment (initialized to -1)
    """
    features = np.full(len(segment), -1, dtype=np.int32)
    
    # Use the stored indices directly
    peak_indices = seg_peaks['index']
    features[peak_indices] = peak_features
    
    return features
