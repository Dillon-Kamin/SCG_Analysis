import os
import numpy as np
import polars as pl
from typing import List, Optional, Tuple
from scipy.signal import butter, sosfiltfilt, find_peaks
from pipeline.data_io import read_device_csv


def filter_signal(
    filepath: str,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
    save: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[pl.DataFrame, Optional[str]]:
    """
    Read, normalize, and bandpass filter z-axis accelerometer data.
    
    Args:
        filepath: Path to raw CSV file
        fs: Sampling frequency in Hz
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order (final order will be 2*order for bandpass)
        save: If True, save filtered signal
        output_dir: Directory to save filtered data. If None, uses ../data/filtered
        
    Returns:
        DataFrame with column ['z'] containing filtered signal
        Path to saved file if save is True, else None
    """
    # Read raw data
    df = read_device_csv(filepath=filepath, columns=["z"])
    z = df["z"].to_numpy()

    # Normalize
    NORMALIZATION_FACTOR = 262144
    z = z / NORMALIZATION_FACTOR

    # Validate filter parameters
    if lowcut >= highcut:
        raise ValueError(f"Invalid band: lowcut {lowcut} >= highcut {highcut}")
    if highcut >= fs / 2:
        raise ValueError(f"highcut {highcut} >= Nyquist ({fs/2})")

    # Apply bandpass filter
    sos = butter(order/2, [lowcut, highcut], btype="band", fs=fs, output="sos")
    z_filt = sosfiltfilt(sos, z)

    # Save if requested
    output_path = None
    if save:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/filtered")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(filepath))
        df_save = pl.DataFrame({"z": z_filt})
        df_save.write_csv(output_path)

    return pl.DataFrame({"z": z_filt}), output_path

def _refine_ao_peak(
    z: np.ndarray,
    first_pass_peak: int,
    local_search_radius: int = 25,
    local_distance: int = 5,
    similarity_threshold: float = 0.6,
) -> int:
    """
    Refine a first-pass detected peak to land on the AO downpeak rather than MV.

    The first-pass peak detection (with distance ~ IBI) reliably finds one peak
    per beat, but may land on either the AO or MV downpeak when their amplitudes
    are similar. This function performs a second-pass detection within a small
    symmetric window around the first-pass peak to find up to three local
    downpeaks, then:

      1. Takes the top 2 by absolute amplitude from those candidates.
      2. Checks whether they are similar enough in amplitude to be a genuine
         AO/MV pair (ratio of smaller to larger >= similarity_threshold).
      3. If yes, returns the earlier of the two (AO always precedes MV).
      4. If no (only one dominant peak, or candidates too dissimilar), returns
         the original first-pass peak unchanged.

    Args:
        z: Full filtered signal array.
        first_pass_peak: Peak index from the first-pass find_peaks call.
        local_search_radius: Half-width of the symmetric local search window
            in samples. Default 25 gives a 50-sample window, safely spanning
            the AO-MV gap at 400-500 Hz.
        local_distance: Minimum distance (samples) between peaks in the second-
            pass find_peaks call. Small (default 5) to allow detection of the
            closely-spaced AO/MV pair.
        similarity_threshold: Minimum ratio of smaller to larger peak amplitude
            (both taken as absolute values) to treat two peaks as a genuine
            AO/MV pair. Default 0.6.

    Returns:
        Refined peak index in global signal coordinates. Falls back to
        first_pass_peak if the window is out of bounds, fewer than 2 peaks are
        found, or the top 2 peaks fail the similarity check.
    """
    window_start = first_pass_peak - local_search_radius
    window_end   = first_pass_peak + local_search_radius

    # Bail out if window extends beyond signal boundaries
    if window_start < 0 or window_end > len(z):
        return first_pass_peak

    local_signal = z[window_start:window_end]

    # Second-pass: find downpeaks within local window, no prominence filter —
    # let the similarity check do the discrimination
    local_peaks, _ = find_peaks(-local_signal, distance=local_distance)

    if len(local_peaks) < 2:
        return first_pass_peak

    # Convert to global indices
    global_peaks = local_peaks + window_start

    # Sort candidates by absolute amplitude (largest first), take top 3
    sorted_by_amp = sorted(global_peaks, key=lambda p: abs(z[p]), reverse=True)
    top_candidates = sorted_by_amp[:3]

    # Apply similarity check to the top 2
    top_two = sorted(top_candidates[:2], key=lambda p: abs(z[p]), reverse=True)
    amp_larger  = abs(z[top_two[0]])
    amp_smaller = abs(z[top_two[1]])

    if amp_larger == 0:
        return first_pass_peak

    if amp_smaller / amp_larger >= similarity_threshold:
        # Genuine AO/MV pair — return the earlier one (AO precedes MV)
        return min(top_two)
    else:
        # Only one dominant peak — original first-pass peak is reliable
        return first_pass_peak

def segment_signal(
    df: Optional[pl.DataFrame] = None,
    filepath: Optional[str] = None,
    fs: float = 250,
    distance: int = 150,
    prominence: float = 0.01,
    tolerance: float = 1.5,
    segment_width: int = 150,
    averaging_window: int = 10,
    ao_mv_refinement: bool = True,
    local_search_radius: int = 25,
    local_distance: int = 5,
    similarity_threshold: float = 0.6,
    save: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Segment filtered signal around downward peaks (beats).

    Uses a two-pass peak detection strategy to robustly anchor segments on the
    AO downpeak rather than the MV downpeak when the two are similar in amplitude:

      Pass 1 — Standard find_peaks with distance ~ IBI. Finds one candidate per
               beat. May land on AO or MV when their amplitudes are close.

      Pass 2 — For each first-pass peak, searches a small symmetric window
               (local_search_radius samples either side) with a tight distance
               constraint to detect both AO and MV candidates. If the top 2
               peaks are sufficiently similar in amplitude (>= similarity_threshold),
               the earlier one is chosen as the AO anchor. Otherwise the original
               first-pass peak is kept.

    This approach is robust to variable device sampling rates (400-500 Hz) because
    it discriminates AO from MV by temporal ordering rather than amplitude or
    fixed sample-count timing.

    Args:
        df: Polars DataFrame with 'z' column (filtered signal). Provide either df or filepath.
        filepath: Path to CSV with filtered signal (column 'z'). Alternative to df.
        fs: Sampling frequency (Hz).
        distance: Minimum distance between peaks in first-pass detection (samples).
        prominence: Minimum prominence of detected peaks in first-pass detection.
        tolerance: IQR multiplier for outlier rejection of first-pass peak amplitudes.
        segment_width: Width (in samples) for each segment (half width applied around peak).
        averaging_window: If >0, average each segment with its neighbors.
        ao_mv_refinement: If True, apply second-pass AO/MV anchor refinement.
            Set False to reproduce original behaviour exactly.
        local_search_radius: Half-width of the local search window for second-pass
            detection (samples). Default 25 gives a 50-sample window.
        local_distance: Minimum distance between peaks in second-pass detection
            (samples). Default 5 allows detection of closely-spaced AO/MV pairs.
        similarity_threshold: Minimum amplitude ratio (smaller/larger) for two
            local peaks to be treated as a genuine AO/MV pair. Default 0.6.
        save: If True, save segments to .npy file.
        output_dir: Directory to save segments. If None, uses ../data/segmented

    Returns:
        np.ndarray: Array of segmented beats, shape (n_segments, segment_width).
        Path to saved file if save is True, else None
    """
    # Load signal
    if df is not None:
        z = df["z"].to_numpy()
    elif filepath is not None:
        z = read_device_csv(filepath=filepath, columns=["z"])["z"].to_numpy()
    else:
        raise ValueError("Either df or filepath must be provided.")

    output_path = None

    # --- Pass 1: one candidate peak per beat ---
    peaks, _ = find_peaks(-z, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        return np.array([]), output_path

    # Outlier removal on first-pass amplitudes (rejects genuinely bad beats)
    peak_vals = z[peaks]
    q1, q3 = np.percentile(peak_vals, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - iqr * tolerance, q3 + iqr * tolerance
    filtered_peaks = [p for p in peaks if lower <= z[p] <= upper]

    if len(filtered_peaks) == 0:
        return np.array([]), output_path

    # --- Pass 2: refine each peak to land on AO rather than MV ---
    if ao_mv_refinement:
        filtered_peaks = [
            _refine_ao_peak(
                z, p,
                local_search_radius=local_search_radius,
                local_distance=local_distance,
                similarity_threshold=similarity_threshold
            )
            for p in filtered_peaks
        ]

    # --- Segmentation around refined anchors ---
    half_width = segment_width // 2
    segments = np.array([
        z[p - half_width:p + half_width]
        for p in filtered_peaks
        if p - half_width >= 0 and p + half_width <= len(z)
    ])

    if len(segments) == 0:
        return np.array([]), output_path

    # Optional smoothing via neighbor averaging
    if averaging_window > 0 and len(segments) > 2 * averaging_window:
        new_segments = []
        for i in range(averaging_window, len(segments) - averaging_window):
            start = i - averaging_window
            end = i + averaging_window + 1
            avg_seg = np.mean(segments[start:end], axis=0)
            new_segments.append(avg_seg)
        segments = np.array(new_segments)

    # Save if requested
    if save and filepath is not None:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/segmented")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(filepath).replace(".csv", "")
        output_path = os.path.join(output_dir, f"{base_name}_segments.npy")
        np.save(output_path, segments)

    return segments, output_path

def create_reference_signal(
    segments: Optional[np.ndarray] = None,
    filepath: Optional[str] = None,
    fs: float = 250,
    output_dir: Optional[str] = None
) -> Tuple[pl.DataFrame, Optional[str]]:
    """
    Generate and save a reference signal from segments.

    Averages all segments into a reference signal, labels 13 features (0-12)
    based on extrema positions, and saves as a CSV.
    
    Features are labeled as:
    - 0: Center downward peak (segmentation anchor point)
    - 1-4: Four extrema to the left (4 is closest to center)
    - 5-12: Eight extrema to the right (5 is closest to center)

    Args:
        segments: Array of segment data, shape (n_segments, segment_width). Provide either segments or filepath.
        filepath: Path to use for loading segments if not provided directly.
        fs: Sampling frequency (Hz).
        output_dir: Directory to save reference CSV. If None, uses ../data/references

    Returns:
        Polars DataFrame with columns: ['Time', 'Amplitude', 'Feature']
        Path to saved reference CSV
    """
    # Load segments if not provided
    if segments is None:
        if filepath is None:
            raise ValueError("Either segments or filepath must be provided.")
        
        segments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/segmented")
        base_name = os.path.basename(filepath).replace(".csv", "")
        segments_path = os.path.join(segments_dir, f"{base_name}_segments.npy")
        
        if not os.path.exists(segments_path):
            raise FileNotFoundError(f"Segments file not found: {segments_path}")
        
        segments = np.load(segments_path)

    if segments is None or len(segments) == 0:
        raise ValueError("No valid segments detected for reference signal.")

    # Average segments
    avg_z = np.mean(segments, axis=0)
    avg_time = np.arange(len(avg_z)) / fs

    # Detect down/up peaks
    down_peaks, _ = find_peaks(-avg_z)
    up_peaks, _ = find_peaks(avg_z)

    # Initialize features (all -1 except labeled points)
    features = np.full(len(avg_z), -1, dtype=int)
    center_idx = len(avg_z) // 2
    features[center_idx] = 0  # Center is always the segmentation downpeak

    # Combine and sort all extrema by position
    all_extrema = sorted(
        [(idx, 'down') for idx in down_peaks] + [(idx, 'up') for idx in up_peaks]
    )

    # Label left features (1-4, with 4 closest to center)
    left_extrema = [idx for idx, _ in all_extrema if idx < center_idx]
    for i, idx in enumerate(reversed(left_extrema[-4:])):
        features[idx] = 4 - i

    # Label right features (5-12)
    right_extrema = [idx for idx, _ in all_extrema if idx > center_idx]
    for i, idx in enumerate(right_extrema[:8]):  # Take up to 8 extrema
        features[idx] = 5 + i

    # Create DataFrame
    df_ref = pl.DataFrame({
        "Time": avg_time,
        "Amplitude": avg_z,
        "Feature": features
    })

    # Save reference
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/references")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if filepath is not None:
        base_name = os.path.basename(filepath).replace("_segments.npy", "")
        output_path = os.path.join(output_dir, f"{base_name}_reference.csv")
    else:
        output_path = os.path.join(output_dir, "reference.csv")
    
    df_ref.write_csv(output_path)
    print(f"Reference signal saved to {output_path}")

    return df_ref, output_path
