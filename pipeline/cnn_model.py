import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SCGDataset(Dataset):
    """
    Dataset of raw SCG beat segments for 1D CNN classification.

    Loads aligned segment tuples, applies an absolute score threshold to
    filter low-quality beats, crops each segment asymmetrically around the
    AO anchor (discarding uninformative pre-AO samples), and returns a
    (waveform, label) pair.

    Args:
        segments:       List of (time, amplitude, features, score, ref_idx) tuples.
        label:          Integer class label for all segments in this dataset.
        score_threshold: Absolute upper bound on alignment score. Segments with
                         score >= threshold are discarded. None keeps all segments.
        crop_start:     First sample index to keep (samples before this are dropped).
                        Default 100 discards the left third of a 300-sample segment,
                        keeping the physiologically rich post-AO region.
        crop_end:       Last sample index (exclusive). Default 300 keeps everything
                        to the right of crop_start.
    """
    def __init__(
        self,
        segments: List[Tuple],
        label: int,
        score_threshold: Optional[float] = None,
        crop_start: int = 100,
        crop_end: int = 300,
    ):
        self.label = label
        self.crop_start = crop_start
        self.crop_end = crop_end

        self.waveforms = []
        for time, amplitude, features, score, ref_idx in segments:
            if score_threshold is not None and score >= score_threshold:
                continue
            crop = amplitude[crop_start:crop_end].astype(np.float32)
            if len(crop) == 0:
                continue
            self.waveforms.append(crop)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # Shape: (1, crop_width) — channel-first for Conv1d
        x = torch.tensor(self.waveforms[idx]).unsqueeze(0)
        y = torch.tensor(self.label, dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SCGConv1D(nn.Module):
    """
    Small 1D convolutional network for SCG beat classification.

    Architecture:
        3 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
        followed by two fully connected layers with dropout.

    Batch normalisation handles session-to-session amplitude variability.
    Dropout at both FC layers guards against overfitting on small datasets.
    L2 regularisation is applied via weight_decay in the optimizer.

    Args:
        input_width: Number of samples in the cropped input waveform.
        n_classes:   Number of output classes.
        dropout_conv: Dropout probability after flattening conv output.
        dropout_fc:   Dropout probability after first FC layer.
    """
    def __init__(
        self,
        input_width: int,
        n_classes: int = 2,
        dropout_conv: float = 0.5,
        dropout_fc: float = 0.3,
    ):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # width / 2

            # Block 2
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # width / 4

            # Block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                          # width / 8
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_width)
            flat_size = self.conv_blocks(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_conv),
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv_blocks(x))


# ---------------------------------------------------------------------------
# Score threshold calibration
# ---------------------------------------------------------------------------

def calibrate_score_threshold(
    reference_aligned_path: str,
    percentile: float = 75.0,
) -> float:
    """
    Derive an absolute alignment score threshold from a reference file.

    Computes the given percentile of alignment scores from the reference
    recording (which should be a clean, well-aligned file). This value is
    saved alongside the model and applied consistently at both training and
    prediction time, replacing the per-file relative cutoff used previously.

    Args:
        reference_aligned_path: Path to .npy aligned segments from the
                                 reference recording.
        percentile: Score percentile to use as cutoff. Default 75 keeps
                    the best 75% of the reference file's beats as the
                    quality bar.

    Returns:
        Absolute score threshold (float). Beats with score >= this value
        will be discarded.
    """
    results = list(np.load(reference_aligned_path, allow_pickle=True))
    scores = np.array([float(score) for _, _, _, score, _ in results])
    threshold = float(np.percentile(scores, percentile))
    print(f"Score threshold calibrated at {percentile}th percentile: {threshold:.4f}")
    return threshold


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cnn_model(
    training_data: Dict[str, List[str]],
    reference_aligned_path: str,
    score_percentile: float = 75.0,
    crop_start: int = 0,
    crop_end: int = 300,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout_conv: float = 0.5,
    dropout_fc: float = 0.3,
    val_split: float = 0.15,
    random_state: int = 42,
    save_model: bool = True,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[SCGConv1D, Optional[str]]:
    """
    Train a 1D CNN on raw SCG beat waveforms.

    Quality filtering uses an absolute score threshold derived from the
    reference file rather than a per-file percentile, ensuring consistent
    filtering between training and prediction.

    Class imbalance is handled via WeightedRandomSampler so the model sees
    balanced batches regardless of unequal segment counts per class.

    Args:
        training_data:          Dict mapping class name to list of aligned .npy paths.
        reference_aligned_path: Path to aligned .npy for the reference recording.
                                 Used to calibrate the absolute score threshold.
        score_percentile:       Percentile of reference scores to use as threshold.
        crop_start:             First sample to keep from each segment (pre-AO discard).
        crop_end:               Last sample to keep (exclusive).
        n_epochs:               Training epochs.
        batch_size:             Batch size.
        learning_rate:          Adam learning rate.
        weight_decay:           L2 regularisation strength.
        dropout_conv:           Dropout after conv flatten.
        dropout_fc:             Dropout after first FC layer.
        val_split:              Fraction of training data held out for validation.
        random_state:           Seed for reproducibility.
        save_model:             If True, save model and metadata to disk.
        model_path:             Full path for saved model. Auto-generated if None.
        output_dir:             Directory for model. Defaults to ../models.

    Returns:
        Trained SCGConv1D model, path to saved model file (or None).
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Calibrate absolute score threshold from reference file
    score_threshold = calibrate_score_threshold(reference_aligned_path, score_percentile)

    class_names = list(training_data.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    input_width = crop_end - crop_start

    print(f"\nClasses: {class_names}")
    print(f"Input width (samples): {input_width}  [{crop_start}:{crop_end}]")

    # Build full dataset per class
    all_waveforms = []
    all_labels = []

    for class_name, paths in training_data.items():
        class_idx = class_to_idx[class_name]
        class_count = 0

        for path in paths:
            if not os.path.exists(path):
                print(f"  Warning: not found: {path}, skipping")
                continue

            results = list(np.load(path, allow_pickle=True))
            ds = SCGDataset(
                results, class_idx,
                score_threshold=score_threshold,
                crop_start=crop_start,
                crop_end=crop_end,
            )
            for waveform, _ in ds:
                all_waveforms.append(waveform)
                all_labels.append(class_idx)
            class_count += len(ds)
            print(f"  {class_name}: {len(ds)} segments from {os.path.basename(path)}")

        print(f"  → {class_name} total: {class_count}")

    if len(all_waveforms) == 0:
        raise ValueError("No training segments loaded. Check file paths and score threshold.")

    all_waveforms = torch.stack(all_waveforms)   # (N, 1, input_width)
    all_labels    = torch.tensor(all_labels, dtype=torch.long)

    # Train / val split
    n_total = len(all_labels)
    n_val   = max(1, int(n_total * val_split))
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(random_state))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = all_waveforms[train_idx], all_labels[train_idx]
    X_val,   y_val   = all_waveforms[val_idx],   all_labels[val_idx]

    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}")

    # Weighted sampler for class balance
    class_counts = torch.bincount(y_train)
    weights = 1.0 / class_counts.float()
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        list(zip(X_train, y_train)),
        batch_size=batch_size,
        sampler=sampler,
    )
    val_loader = DataLoader(
        list(zip(X_val, y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    # Model, loss, optimizer
    model = SCGConv1D(
        input_width=input_width,
        n_classes=len(class_names),
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nTraining for {n_epochs} epochs...")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 50)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, n_epochs + 1):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total   += len(yb)

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                val_loss    += loss.item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(yb)

        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total
        v_loss = val_loss / val_total
        v_acc  = val_correct / val_total

        scheduler.step(v_loss)

        print(f"{epoch:>6}  {t_loss:>10.4f}  {t_acc:>9.3f}  {v_loss:>8.4f}  {v_acc:>7.3f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    output_path = None
    if save_model:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
        os.makedirs(output_dir, exist_ok=True)
        if model_path is None:
            model_path = os.path.join(output_dir, "cnn_model.pkl")

        model_data = {
            'model_state':      model.state_dict(),
            'class_names':      class_names,
            'class_to_idx':     class_to_idx,
            'input_width':      input_width,
            'crop_start':       crop_start,
            'crop_end':         crop_end,
            'score_threshold':  score_threshold,
            'dropout_conv':     dropout_conv,
            'dropout_fc':       dropout_fc,
        }
        joblib.dump(model_data, model_path)
        output_path = model_path
        print(f"Model saved to {model_path}")

    return model, output_path


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_cnn_model(
    model_path: str,
    aligned_data: Optional[List[Tuple]] = None,
    aligned_path: Optional[str] = None,
) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run beat-by-beat prediction using a trained CNN.

    Uses the absolute score threshold saved at training time — no per-file
    percentile recomputation.

    Args:
        model_path:    Path to saved model .pkl file.
        aligned_data:  List of aligned segment tuples (from align_segments).
                       Provide either this or aligned_path.
        aligned_path:  Path to .npy of aligned segments. Alternative to aligned_data.

    Returns:
        Tuple of (most_common_class_idx, all_predictions, probabilities).
        probabilities shape: (n_segments, n_classes).
    """
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return (None, None, None)

    model_data = joblib.load(model_path)
    class_names     = model_data['class_names']
    input_width     = model_data['input_width']
    crop_start      = model_data['crop_start']
    crop_end        = model_data['crop_end']
    score_threshold = model_data['score_threshold']
    dropout_conv    = model_data['dropout_conv']
    dropout_fc      = model_data['dropout_fc']

    print(f"Model loaded. Classes: {class_names}")
    print(f"Score threshold (absolute): {score_threshold:.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SCGConv1D(
        input_width=input_width,
        n_classes=len(class_names),
        dropout_conv=dropout_conv,
        dropout_fc=dropout_fc,
    ).to(device)
    model.load_state_dict(model_data['model_state'])
    model.eval()

    # Load results
    if aligned_data is not None:
        results = aligned_data
    elif aligned_path is not None:
        results = list(np.load(aligned_path, allow_pickle=True))
    else:
        raise ValueError("Either aligned_data or aligned_path must be provided.")

    # Build dataset using saved absolute threshold
    ds = SCGDataset(
        results, label=0,          # label unused at prediction time
        score_threshold=score_threshold,
        crop_start=crop_start,
        crop_end=crop_end,
    )

    if len(ds) == 0:
        print("No segments passed the score threshold.")
        return (None, None, None)

    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb     = xb.to(device)
            logits = model(xb)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    predictions  = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    most_common  = int(np.bincount(predictions).argmax())

    print(f"\nPrediction Summary ({len(predictions)} segments):")
    for idx, name in enumerate(class_names):
        count = np.sum(predictions == idx)
        pct   = count / len(predictions) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    print(f"\nMost common: {class_names[most_common]}")

    return most_common, predictions, probabilities
