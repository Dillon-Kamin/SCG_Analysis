import os
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Tuple, Optional


# Features to include. 11 and 12 excluded as they are in the unreliable
# far-right tail of the segment and add noise without physiological benefit.
ACTIVE_FEATURES = list(range(0, 11))  # 0-10 inclusive


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

    For each segment, extracts four groups of features:

    1. Reference amplitude
       - 'Feature_0_amp': Amplitude of the AO downpeak (Feature 0).

    2. Vector features (Feature 0 → Feature N, for N in 1-10)
       - 'Vector_N_dt':   Time difference from Feature 0 to Feature N (seconds).
                          Negative for features to the left of AO (1-4).
       - 'Vector_N_damp': Amplitude difference from Feature 0 to Feature N.
       Kept as separate components rather than polar coordinates to avoid
       mixing units (seconds vs amplitude), which would make angles sensitive
       to sampling rate and sensor gain rather than physiology.

    3. Interval features (pairwise time differences, excluding pairs with Feature 0)
       - 'Interval_I_J_dt': Time from Feature I to Feature J (seconds),
                             for all unique pairs (I, J) where I < J and I ≠ 0,
                             across features 1-10. Gives 45 interval columns.
                             (Pairs involving 0 are excluded as they duplicate Vector features)

    4. Pairwise amplitude difference features (same pair set as group 3)
       - 'Amp_I_J_damp': Amplitude of Feature J minus amplitude of Feature I,
                          for all unique pairs (I, J) where I < J and I ≠ 0.
                          Gives 45 amplitude difference columns.
       Together with group 3, these encode inter-feature slope information
       (e.g. the rise from Feature 6 to Feature 7) without dividing units.
       The RF can combine dt and damp implicitly to learn slope relationships,
       or use them independently if one dimension is more informative.

    Total columns: 1 + 20 + 45 + 45 = 111.

    Only segments with alignment scores below the percentile cutoff are
    included. Segments without Feature 0 are skipped.

    Args:
        aligned_results: Output from align_segments() as list of tuples:
                        (time, amplitude, features, score, ref_idx)
        aligned_path: Path to .npy file containing aligned results.
                      Alternative to aligned_results.
        score_percentile_cutoff: Only include segments with scores below
                                 this percentile. Lower scores are better.
                                 Default 75 = best 75%.
        save: If True, save features to CSV.
        output_dir: Directory for saving. If None, uses ../data/features
        filename: Name for saved file. If None, uses 'features.csv'

    Returns:
        DataFrame with columns described above.
        Path to saved CSV if save=True, else None.
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

    # Score cutoff
    scores = np.array([score for _, _, _, score, _ in results])
    cutoff = np.percentile(scores, score_percentile_cutoff)

    # Pre-compute all pairwise combinations once
    # Exclude pairs (0, X) since those are already in Vector features
    pair_combinations = [(i, j) for i, j in combinations(ACTIVE_FEATURES, 2) if i != 0]

    feature_rows = []

    for time, amplitude, feature_array, score, ref_idx in results:
        if score >= cutoff:
            continue

        # Collect active labeled points
        feature_mask = feature_array >= 0
        if not np.any(feature_mask):
            continue

        feature_indices = np.where(feature_mask)[0]
        feature_times  = time[feature_indices]
        feature_amps   = amplitude[feature_indices]
        feature_labels = feature_array[feature_indices]

        # Must have Feature 0
        ref_mask = feature_labels == 0
        if not np.any(ref_mask):
            continue

        ref_time = feature_times[ref_mask][0]
        ref_amp  = feature_amps[ref_mask][0]

        # Build a lookup {feature_num: (time, amp)} for active features only
        feat_lookup = {}
        for ft, fa, fl in zip(feature_times, feature_amps, feature_labels):
            fl_int = int(fl)
            if fl_int in ACTIVE_FEATURES:
                feat_lookup[fl_int] = (float(ft), float(fa))

        # --- Group 1: reference amplitude ---
        feature_dict = {'Feature_0_amp': ref_amp}

        # --- Group 2: vector features (Feature 0 → Feature N) ---
        for feat_num in ACTIVE_FEATURES:
            if feat_num == 0:
                continue
            if feat_num not in feat_lookup:
                continue
            ft, fa = feat_lookup[feat_num]
            feature_dict[f'Vector_{feat_num}_dt']   = ft - ref_time
            feature_dict[f'Vector_{feat_num}_damp'] = fa - ref_amp

        # --- Groups 3 & 4: pairwise interval and amplitude differences ---
        for i, j in pair_combinations:
            if i not in feat_lookup or j not in feat_lookup:
                continue
            t_i, a_i = feat_lookup[i]
            t_j, a_j = feat_lookup[j]
            feature_dict[f'Interval_{i}_{j}_dt']   = t_j - t_i
            feature_dict[f'Amp_{i}_{j}_damp']       = a_j - a_i

        feature_rows.append(feature_dict)

    if len(feature_rows) == 0:
        raise ValueError("No valid features extracted. Check score cutoff or alignment quality.")

    df = pd.DataFrame(feature_rows)

    # --- Column ordering ---
    base_cols     = ['Feature_0_amp']
    vector_cols   = []
    for i in ACTIVE_FEATURES:
        if i == 0:
            continue
        vector_cols += [f'Vector_{i}_dt', f'Vector_{i}_damp']

    interval_cols = [f'Interval_{i}_{j}_dt' for i, j in pair_combinations]
    amp_diff_cols = [f'Amp_{i}_{j}_damp'    for i, j in pair_combinations]

    ordered_cols = [
        c for c in base_cols + vector_cols + interval_cols + amp_diff_cols
        if c in df.columns
    ]
    df = df[ordered_cols]

    # Save if requested
    output_path = None
    if save:
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../data/features"
            )
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