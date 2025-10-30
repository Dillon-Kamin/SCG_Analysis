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


def segment_signal(
    df: Optional[pl.DataFrame] = None,
    filepath: Optional[str] = None,
    fs: float = 500,
    distance: int = 150,
    prominence: float = 0.01,
    tolerance: float = 1.5,
    segment_width: int = 150,
    averaging_window: int = 10,
    save: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Segment filtered signal around downward peaks (beats).

    Args:
        df: Polars DataFrame with 'z' column (filtered signal). Provide either df or filepath.
        filepath: Path to CSV with filtered signal (column 'z'). Alternative to df.
        fs: Sampling frequency (Hz).
        distance: Minimum distance between peaks (samples).
        prominence: Minimum prominence of detected peaks.
        tolerance: Stddev multiplier for outlier rejection.
        segment_width: Width (in samples) for each segment (half width applied around peak).
        averaging_window: If >0, average each segment with its neighbors.
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

    # Detect downward peaks
    peaks, _ = find_peaks(-z, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        return np.array([]), output_path

    # Outlier removal
    peak_vals = z[peaks]
    q1, q3 = np.percentile(peak_vals, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - iqr * tolerance, q3 + iqr * tolerance
    filtered_peaks = [p for p in peaks if lower <= z[p] <= upper]

    if len(filtered_peaks) == 0:
        return np.array([]), output_path

    # Segment around peaks
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
    fs: float = 500,
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
