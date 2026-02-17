"""
Clustering helper script for SCG_Analysis

Features:
- Produces vectorized features per-segment and clusters them
- Optional PCA before clustering
- Flexible visualization: side-by-side, separate plots, or overlay
- Color by labels or filenames for comparison
- Elbow plot generation to help determine optimal k

Usage:
1. Configure parameters directly in script and run:
   python cluster.py

2. Or use command-line arguments:
   python cluster.py --files data/raw/*.csv --references data/ref.csv --n-clusters 3
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse
from pipeline.preprocess import filter_signal, segment_signal
from pipeline.alignment import align_segments
from pipeline.feature_extraction import extract_vectorized_features


# ============================================================================
# CONFIGURATION SECTION - Edit these parameters for direct execution
# ============================================================================

CONFIG = {
    # Input files
    'files': [
        'data/raw/engage1.csv',
        'data/raw/engage2.csv',
        'data/raw/engage3.csv',
        'data/raw/engage4.csv',
        'data/raw/engage5.csv',
        'data/raw/relax4.csv',
        'data/raw/relax5.csv',
    ],
    'references': [
        'data/references/engage5_reference.csv',
    ],
    
    # Labels mapping (filename -> label) - set to None if using filenames directly
    'labels': {
        'engage1.csv': 'engaged1',
        'engage2.csv': 'engaged',
        'engage3.csv': 'engaged',
        'engage4.csv': 'engaged',
        'engage5.csv': 'engaged5',
        'relax1.csv': 'relaxed',
        'relax2.csv': 'relaxed',
        'relax3.csv': 'relaxed',
        'relax4.csv': 'relaxed',
        'relax5.csv': 'relaxed',
    },
    
    # Clustering parameters
    'algorithm': 'kmeans',  # 'kmeans' or 'dbscan'
    'n_clusters': 2,  # Only used for kmeans
    'dbscan_eps': 0.5,  # Only used for dbscan
    'dbscan_min_samples': 5,  # Only used for dbscan
    
    # PCA settings
    'use_pca': False,  # Apply PCA before clustering?
    'pca_components': 10,  # Number of components if use_pca=True
    
    # Visualization settings
    'visualization_mode': 'side_by_side',  # 'side_by_side', 'separate', or 'overlay'
    'comparison_coloring': 'labels',  # 'labels' or 'filenames' - what to show in comparison plot
    'save_cluster_plot': 'figures/clusters.png',  # Path for cluster plot (always generated)
    'save_comparison_plot': 'figures/comparison.png',  # Path for comparison plot (labels/filenames)
    'save_elbow': None,
    'max_clusters_elbow': 15,
    
    # Pipeline parameters
    'fs': 500,
    'lowcut': 2.0,
    'highcut': 50.0,
    'filter_order': 2,
    'dp_distance': 167,
    'dp_prominence': 0.02,
    'dp_tolerance': 1.5,
    'segment_width': 300,
    'averaging_window': 10,
    'score_percentile_cutoff': 75.0,
}

# ============================================================================
# PIPELINE FUNCTIONS
# ============================================================================

def prepare_features_from_files(
    filepaths: List[str],
    reference_paths: List[str],
    pipeline_params: Dict,
    temp_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run filter -> segment -> align -> extract_vectorized_features for each file.
    
    Returns a DataFrame with one row per segment, including:
    - Feature columns (numeric)
    - 'source_file': original filename (basename)
    - 'label': label (if provided later)
    """
    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    all_feature_dfs = []

    for fp in filepaths:
        print(f"Processing: {fp}")

        # Filter
        df_filt, _ = filter_signal(
            filepath=fp,
            fs=pipeline_params['fs'],
            lowcut=pipeline_params['lowcut'],
            highcut=pipeline_params['highcut'],
            order=pipeline_params['filter_order'],
            save=False
        )

        # Segment
        segments, seg_path = segment_signal(
            df=df_filt,
            filepath=fp,
            fs=pipeline_params['fs'],
            distance=pipeline_params['dp_distance'],
            prominence=pipeline_params['dp_prominence'],
            tolerance=pipeline_params['dp_tolerance'],
            segment_width=pipeline_params['segment_width'],
            averaging_window=pipeline_params['averaging_window'],
            save=True,
            output_dir=temp_dir
        )

        if len(segments) == 0 or seg_path is None:
            print(f"  ✗ No segments for {fp}, skipping")
            continue

        # Align
        aligned_results, _ = align_segments(
            segments_path=seg_path,
            reference_paths=reference_paths,
            fs=pipeline_params['fs'],
            save=False
        )

        if len(aligned_results) == 0:
            print(f"  ✗ Alignment failed for {fp}, skipping")
            continue

        # Extract features
        try:
            feats_df, _ = extract_vectorized_features(
                aligned_results=aligned_results,
                score_percentile_cutoff=pipeline_params['score_percentile_cutoff'],
                save=False
            )
        except Exception as e:
            print(f"  ✗ Feature extraction failed for {fp}: {e}")
            continue

        if feats_df is None or len(feats_df) == 0:
            print(f"  ✗ No features for {fp}, skipping")
            continue

        # Add metadata
        feats_df = feats_df.copy()
        feats_df['source_file'] = os.path.basename(fp)
        all_feature_dfs.append(feats_df)

    if len(all_feature_dfs) == 0:
        raise ValueError("No features created from input files.")

    combined = pd.concat(all_feature_dfs, ignore_index=True)
    return combined


def prepare_feature_matrix(
    feats: pd.DataFrame,
    use_pca: bool = False,
    n_components: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, StandardScaler, Optional[PCA]]:
    """
    Prepare feature matrix for clustering:
    1. Extract feature columns (drop metadata)
    2. Fill NaNs with column means
    3. Standardize
    4. Optionally apply PCA
    
    Returns (X_processed, scaler, pca_model)
    """
    meta_cols = ['source_file', 'label', 'filename_group']
    feature_cols = [c for c in feats.columns if c not in meta_cols]
    
    X = feats[feature_cols].copy()
    X = X.fillna(X.mean())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    
    pca_model = None
    if use_pca:
        # Ensure we don't request more components than features
        n_comp = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
        pca_model = PCA(n_components=n_comp, random_state=random_state)
        X_processed = pca_model.fit_transform(X_scaled)
        print(f"PCA: {X_scaled.shape[1]} features -> {n_comp} components")
        print(f"Explained variance: {pca_model.explained_variance_ratio_.sum():.3f}")
    else:
        X_processed = X_scaled
    
    return X_processed, scaler, pca_model


def cluster_features(
    X: np.ndarray,
    algorithm: str = 'kmeans',
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Cluster the feature matrix using specified algorithm.
    
    Returns cluster labels.
    """
    if algorithm == 'kmeans':
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        print(f"KMeans clustering with k={n_clusters}")
    elif algorithm == 'dbscan':
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"DBSCAN clustering: {n_clusters_found} clusters, {n_noise} noise points")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return labels


def plot_elbow(
    X: np.ndarray,
    max_clusters: int = 15,
    save_path: Optional[str] = None,
    random_state: int = 42
):
    """
    Generate elbow plot for KMeans to help choose optimal k.
    """
    max_k = min(max_clusters, X.shape[0] - 1)
    inertias = []
    K_range = range(1, max_k + 1)
    
    print(f"Generating elbow plot for k=1 to k={max_k}...")
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Elbow plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_single_scatter(
    ax: plt.Axes,
    X: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    title: str,
    legend_title: str,
    use_pca: bool = False
):
    """
    Helper function to create a single scatter plot.
    """
    x = X[:, 0]
    y = X[:, 1]
    
    unique_labels = sorted(np.unique(labels))
    cmap = plt.get_cmap('tab10')
    
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        display_name = label_names[lbl] if lbl < len(label_names) else f'Unknown {lbl}'
        ax.scatter(x[mask], y[mask], label=display_name, alpha=0.7, s=50,
                  c=[cmap(i % 10)], edgecolors='black', linewidth=0.5)
    
    xlabel = 'PC1' if use_pca else 'Dimension 1'
    ylabel = 'PC2' if use_pca else 'Dimension 2'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title=legend_title, fontsize=10)
    ax.grid(True, alpha=0.2)


def plot_clusters_side_by_side(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    comparison_labels: np.ndarray,
    comparison_names: List[str],
    comparison_type: str,
    use_pca: bool = False,
    save_path: Optional[str] = None
):
    """
    Create side-by-side comparison plot.
    
    Parameters:
    - comparison_labels: Numeric labels for comparison (e.g., label indices or filename indices)
    - comparison_names: Human-readable names for comparison labels
    - comparison_type: 'labels' or 'filenames'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Cluster assignments
    cluster_names = [f'Cluster {i}' if i != -1 else 'Noise' 
                     for i in sorted(np.unique(cluster_labels))]
    plot_single_scatter(ax1, X, cluster_labels, cluster_names, 
                       'Clustering Results', 'Cluster', use_pca)
    
    # Right: Comparison (labels or filenames)
    title = 'Ground Truth Labels' if comparison_type == 'labels' else 'Source Files'
    legend_title = 'Label' if comparison_type == 'labels' else 'File'
    plot_single_scatter(ax2, X, comparison_labels, comparison_names,
                       title, legend_title, use_pca)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Side-by-side plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_clusters_separate(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    comparison_labels: np.ndarray,
    comparison_names: List[str],
    comparison_type: str,
    use_pca: bool = False,
    cluster_save_path: Optional[str] = None,
    comparison_save_path: Optional[str] = None
):
    """
    Create two separate plots: one for clusters, one for comparison.
    """
    # Plot 1: Clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_names = [f'Cluster {i}' if i != -1 else 'Noise' 
                     for i in sorted(np.unique(cluster_labels))]
    plot_single_scatter(ax, X, cluster_labels, cluster_names,
                       'Clustering Results', 'Cluster', use_pca)
    plt.tight_layout()
    
    if cluster_save_path:
        plt.savefig(cluster_save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Cluster plot saved to {cluster_save_path}")
        plt.close()
    else:
        plt.show()
    
    # Plot 2: Comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    title = 'Ground Truth Labels' if comparison_type == 'labels' else 'Source Files'
    legend_title = 'Label' if comparison_type == 'labels' else 'File'
    plot_single_scatter(ax, X, comparison_labels, comparison_names,
                       title, legend_title, use_pca)
    plt.tight_layout()
    
    if comparison_save_path:
        plt.savefig(comparison_save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {comparison_save_path}")
        plt.close()
    else:
        plt.show()


def plot_clusters_overlay(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    comparison_labels: np.ndarray,
    comparison_names: List[str],
    comparison_type: str,
    use_pca: bool = False,
    save_path: Optional[str] = None
):
    """
    Overlay plot: colors = comparison (labels/files), shapes = clusters.
    """
    x = X[:, 0]
    y = X[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']
    cmap = plt.get_cmap('tab10')
    
    unique_comparisons = sorted(np.unique(comparison_labels))
    unique_clusters = sorted(np.unique(cluster_labels))
    
    # Plot each combination
    for i, comp_lbl in enumerate(unique_comparisons):
        comp_name = comparison_names[comp_lbl] if comp_lbl < len(comparison_names) else f'Unknown {comp_lbl}'
        for j, cluster_lbl in enumerate(unique_clusters):
            mask = (comparison_labels == comp_lbl) & (cluster_labels == cluster_lbl)
            if mask.sum() > 0:
                cluster_name = f'C{cluster_lbl}' if cluster_lbl != -1 else 'Noise'
                label = f'{comp_name} / {cluster_name}'
                ax.scatter(x[mask], y[mask], label=label, alpha=0.7, s=100,
                          c=[cmap(i % 10)], marker=markers[j % len(markers)],
                          edgecolors='black', linewidth=0.5)
    
    xlabel = 'PC1' if use_pca else 'Dimension 1'
    ylabel = 'PC2' if use_pca else 'Dimension 2'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    title_suffix = 'Labels' if comparison_type == 'labels' else 'Files'
    ax.set_title(f'Overlay: {title_suffix} (color) vs Clusters (shape)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Overlay plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def prepare_comparison_data(
    feats: pd.DataFrame,
    comparison_type: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare comparison labels and names based on comparison type.
    
    Returns:
    - comparison_labels: Numeric array for each sample
    - comparison_names: List of names corresponding to label indices
    """
    if comparison_type == 'labels':
        # Use provided labels
        if 'label' not in feats.columns or feats['label'].isna().all():
            raise ValueError("comparison_type='labels' but no labels provided in config")
        
        unique_labels = sorted(feats['label'].dropna().unique())
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        comparison_labels = feats['label'].map(label_to_idx).fillna(-1).astype(int).values
        comparison_names = unique_labels
        
    elif comparison_type == 'filenames':
        # Use source filenames
        unique_files = sorted(feats['source_file'].unique())
        file_to_idx = {f: i for i, f in enumerate(unique_files)}
        comparison_labels = feats['source_file'].map(file_to_idx).values
        comparison_names = unique_files
        
    else:
        raise ValueError(f"Unknown comparison_type: {comparison_type}")
    
    return comparison_labels, comparison_names


def load_labels_file(filepath: str) -> Dict[str, str]:
    """
    Load a labels CSV or JSON mapping.
    CSV expected format: filename,label (no header)
    
    Returns mapping from filename -> label
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    df = pd.read_csv(filepath, header=None)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ['filename', 'label']
        return dict(zip(df['filename'].astype(str), df['label'].astype(str)))
    else:
        raise ValueError('Unsupported labels file format. Provide CSV (filename,label) or JSON mapping.')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_clustering(config: Dict):
    """
    Main clustering workflow.
    """
    print("=" * 70)
    print("SCG Clustering Analysis")
    print("=" * 70)
    
    # Extract pipeline parameters
    pipeline_params = {
        'fs': config['fs'],
        'lowcut': config['lowcut'],
        'highcut': config['highcut'],
        'filter_order': config['filter_order'],
        'dp_distance': config['dp_distance'],
        'dp_prominence': config['dp_prominence'],
        'dp_tolerance': config['dp_tolerance'],
        'segment_width': config['segment_width'],
        'averaging_window': config['averaging_window'],
        'score_percentile_cutoff': config['score_percentile_cutoff'],
    }
    
    # Process files through pipeline
    print("\n[1/5] Processing files through pipeline...")
    feats = prepare_features_from_files(
        config['files'],
        config['references'],
        pipeline_params
    )
    print(f"✓ Extracted features from {len(feats)} segments")
    
    # Attach labels if provided
    if config.get('labels') is not None:
        feats['label'] = feats['source_file'].map(lambda x: config['labels'].get(x, None))
        print(f"✓ Attached labels to {feats['label'].notna().sum()} segments")
    else:
        feats['label'] = None
    
    # Prepare feature matrix
    print("\n[2/5] Preparing feature matrix...")
    X_cluster, scaler, pca_model = prepare_feature_matrix(
        feats,
        use_pca=config['use_pca'],
        n_components=config['pca_components']
    )
    print(f"✓ Feature matrix shape: {X_cluster.shape}")
    
    # Generate elbow plot
    print("\n[3/5] Generating elbow plot...")
    plot_elbow(
        X_cluster,
        max_clusters=config['max_clusters_elbow'],
        save_path=config.get('save_elbow')
    )
    
    # Perform clustering
    print("\n[4/5] Performing clustering...")
    cluster_labels = cluster_features(
        X_cluster,
        algorithm=config['algorithm'],
        n_clusters=config['n_clusters'],
        eps=config.get('dbscan_eps', 0.5),
        min_samples=config.get('dbscan_min_samples', 5)
    )
    
    print(f"\nCluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} samples")
    
    # Prepare visualization (always use PCA with 2 components for plotting)
    print("\n[5/5] Generating visualizations...")
    pca_viz = PCA(n_components=2, random_state=42)
    
    # Need to go back to scaled features for visualization PCA
    meta_cols = ['source_file', 'label', 'filename_group']
    feature_cols = [c for c in feats.columns if c not in meta_cols]
    X_raw = feats[feature_cols].fillna(feats[feature_cols].mean())
    X_scaled = scaler.transform(X_raw.values)
    X_plot = pca_viz.fit_transform(X_scaled)
    
    # Prepare comparison data
    comparison_type = config.get('comparison_coloring', 'filenames')
    comparison_labels, comparison_names = prepare_comparison_data(feats, comparison_type)
    
    # Generate plots based on visualization mode
    viz_mode = config.get('visualization_mode', 'side_by_side')
    
    if viz_mode == 'side_by_side':
        save_path = config.get('save_cluster_plot') or config.get('save_comparison_plot')
        plot_clusters_side_by_side(
            X_plot,
            cluster_labels,
            comparison_labels,
            comparison_names,
            comparison_type,
            use_pca=True,
            save_path=save_path
        )
        
    elif viz_mode == 'separate':
        plot_clusters_separate(
            X_plot,
            cluster_labels,
            comparison_labels,
            comparison_names,
            comparison_type,
            use_pca=True,
            cluster_save_path=config.get('save_cluster_plot'),
            comparison_save_path=config.get('save_comparison_plot')
        )
        
    elif viz_mode == 'overlay':
        save_path = config.get('save_cluster_plot') or config.get('save_comparison_plot')
        plot_clusters_overlay(
            X_plot,
            cluster_labels,
            comparison_labels,
            comparison_names,
            comparison_type,
            use_pca=True,
            save_path=save_path
        )
    else:
        raise ValueError(f"Unknown visualization_mode: {viz_mode}")
    
    print("\n" + "=" * 70)
    print("Clustering complete!")
    print("=" * 70)
    
    return feats, cluster_labels


def main():
    """
    Entry point supporting both direct execution and CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description='Cluster SCG segments from raw files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use script configuration
  python cluster.py
  
  # Override with CLI arguments
  python cluster.py --files data/raw/*.csv --references data/ref.csv --n-clusters 4
        """
    )
    
    # File inputs
    parser.add_argument('--files', nargs='+', help='Raw CSV files to process')
    parser.add_argument('--references', nargs='+', help='Reference CSV files')
    parser.add_argument('--labels-file', help='CSV or JSON file mapping filename -> label')
    
    # Clustering parameters
    parser.add_argument('--algorithm', choices=['kmeans', 'dbscan'], help='Clustering algorithm')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters for KMeans')
    parser.add_argument('--dbscan-eps', type=float, help='DBSCAN epsilon parameter')
    parser.add_argument('--dbscan-min-samples', type=int, help='DBSCAN min_samples parameter')
    
    # PCA settings
    parser.add_argument('--use-pca', action='store_true', help='Apply PCA before clustering')
    parser.add_argument('--no-pca', action='store_true', help='Do not apply PCA before clustering')
    parser.add_argument('--pca-components', type=int, help='Number of PCA components for clustering')
    
    # Visualization settings
    parser.add_argument('--viz-mode', choices=['side_by_side', 'separate', 'overlay'],
                       help='Visualization mode')
    parser.add_argument('--comparison', choices=['labels', 'filenames'],
                       help='What to show in comparison plot')
    parser.add_argument('--save-cluster-plot', help='Path to save cluster plot')
    parser.add_argument('--save-comparison-plot', help='Path to save comparison plot')
    
    # Output settings
    parser.add_argument('--save-elbow', help='Path to save elbow plot')
    parser.add_argument('--max-clusters-elbow', type=int, help='Max clusters for elbow plot')
    
    # Pipeline parameters
    parser.add_argument('--score-cutoff', type=float, help='Override score_percentile_cutoff')
    
    args = parser.parse_args()
    
    # Start with CONFIG, then override with CLI arguments
    config = CONFIG.copy()
    
    if args.files is not None:
        config['files'] = args.files
    if args.references is not None:
        config['references'] = args.references
    if args.labels_file is not None:
        config['labels'] = load_labels_file(args.labels_file)
    if args.algorithm is not None:
        config['algorithm'] = args.algorithm
    if args.n_clusters is not None:
        config['n_clusters'] = args.n_clusters
    if args.dbscan_eps is not None:
        config['dbscan_eps'] = args.dbscan_eps
    if args.dbscan_min_samples is not None:
        config['dbscan_min_samples'] = args.dbscan_min_samples
    if args.use_pca:
        config['use_pca'] = True
    if args.no_pca:
        config['use_pca'] = False
    if args.pca_components is not None:
        config['pca_components'] = args.pca_components
    if args.viz_mode is not None:
        config['visualization_mode'] = args.viz_mode
    if args.comparison is not None:
        config['comparison_coloring'] = args.comparison
    if args.save_cluster_plot is not None:
        config['save_cluster_plot'] = args.save_cluster_plot
    if args.save_comparison_plot is not None:
        config['save_comparison_plot'] = args.save_comparison_plot
    if args.save_elbow is not None:
        config['save_elbow'] = args.save_elbow
    if args.max_clusters_elbow is not None:
        config['max_clusters_elbow'] = args.max_clusters_elbow
    if args.score_cutoff is not None:
        config['score_percentile_cutoff'] = args.score_cutoff
    
    # Run clustering
    run_clustering(config)


if __name__ == '__main__':
    main()