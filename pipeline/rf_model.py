import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Optional, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pipeline.feature_extraction import extract_vectorized_features, load_features
import warnings
warnings.filterwarnings('ignore')


def train_rf_model(
    training_data: Dict[str, List[str]],
    test_data: Optional[Dict[str, List[str]]] = None,
    score_percentile_cutoff: float = 75.0,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    save_model: bool = True,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[RandomForestClassifier, Optional[str]]:
    """
    Train a Random Forest classifier on aligned segment features.
    
    Args:
        training_data: Dict mapping class names to list of aligned .npy file paths.
                      Example: {'center': ['./data/aligned/center1_aligned.npy'],
                                'left': ['./data/aligned/left1_aligned.npy']}
        test_data: Optional dict with same structure for test set. If None, splits training_data.
        score_percentile_cutoff: Percentile cutoff for feature extraction (best X%).
        test_size: Fraction of data for test set (if test_data is None).
        random_state: Random seed for reproducibility.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees.
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf node.
        save_model: If True, save the trained model.
        model_path: Path to save model. If None, auto-generated.
        output_dir: Directory for model. If None, uses ../models
        
    Returns:
        Trained RandomForestClassifier, path to saved model (if saved).
    """
    # Extract features and labels for training data
    X_train_list = []
    y_train_list = []
    feature_names = None
    
    class_names = list(training_data.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Classes: {class_names}")
    print(f"Extracting features from training data...")
    
    for class_name, paths in training_data.items():
        class_idx = class_to_idx[class_name]
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}, skipping...")
                continue
            
            # Extract features
            features_df, _ = extract_vectorized_features(
                aligned_path=path,
                score_percentile_cutoff=score_percentile_cutoff
            )

            # Get feature names once
            if feature_names is None:
                feature_names = features_df.columns.tolist()
            
            X_train_list.append(features_df.to_numpy())
            y_train_list.append(np.full(len(features_df), class_idx))
            
            print(f"  {class_name}: {len(features_df)} segments from {os.path.basename(path)}")
    
    if len(X_train_list) == 0:
        raise ValueError("No training data loaded. Check file paths.")
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    
    # Handle test data
    if test_data is not None:
        print(f"\nExtracting features from test data...")
        X_test_list = []
        y_test_list = []
        
        for class_name, paths in test_data.items():
            class_idx = class_to_idx[class_name]
            for path in paths:
                if not os.path.exists(path):
                    print(f"Warning: File not found: {path}, skipping...")
                    continue
                
                features_df, _ = extract_vectorized_features(
                    aligned_path=path,
                    score_percentile_cutoff=score_percentile_cutoff
                )
                
                X_test_list.append(features_df.to_numpy())
                y_test_list.append(np.full(len(features_df), class_idx))
                
                print(f"  {class_name}: {len(features_df)} segments from {os.path.basename(path)}")
        
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
    else:
        # Split training data
        print(f"\nSplitting training data: {test_size*100:.0f}% for testing...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
        )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train model
    print(f"\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print(f"\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # ROC AUC for binary classification
    if len(class_names) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        feat_name = feature_names[idx] if feature_names else f"Feature {idx}"
        print(f"  {i}. {feat_name}: {feature_importance[idx]:.4f}")
    
    # Save model
    if save_model:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
        os.makedirs(output_dir, exist_ok=True)
        
        if model_path is None:
            model_path = os.path.join(output_dir, "rf_model.pkl")
        elif not model_path.startswith('/'):
            model_path = os.path.join(output_dir, model_path)
        
        # Save model and class mapping
        model_data = {
            'model': model,
            'class_names': class_names,
            'class_to_idx': class_to_idx,
            'feature_names': feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to {model_path}")
    
    return model, model_path


def predict_rf_model(
    model_path: str,
    aligned_data: Optional[List[Tuple]] = None,
    aligned_path: Optional[str] = None,
    score_percentile_cutoff: float = 75.0,
    return_probabilities: bool = False
) -> Tuple[Optional[int], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Make predictions using a trained Random Forest model.
    
    Args:
        model_path: Path to saved model (.pkl file).
        aligned_data: Output from align_segments(). Provide either this or aligned_path.
        aligned_path: Path to .npy file with aligned segments. Alternative to aligned_data.
        score_percentile_cutoff: Percentile cutoff for feature extraction.
        return_probabilities: If True, return class probabilities for each segment.
        
    Returns:
        Tuple of (most_common_class, all_predictions, probabilities)
        - most_common_class: Most frequently predicted class index
        - all_predictions: Array of predictions for each segment
        - probabilities: Class probabilities (if return_probabilities=True, else None)
    """
    # Load model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return (None, None, None)
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    class_names = model_data['class_names']
    
    print(f"Model loaded. Classes: {class_names}")
    
    # Extract features
    try:
        features_df, _ = extract_vectorized_features(
            aligned_results=aligned_data,
            aligned_path=aligned_path,
            score_percentile_cutoff=score_percentile_cutoff
        )
    except Exception as e:
        print(f"Error extracting features: {e}")
        return (None, None, None)
    
    X = features_df.to_numpy()
    print(f"Predicting on {len(X)} segments...")
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X) if return_probabilities else None
    
    # Most common prediction
    if len(predictions) == 0:
        most_common = None
    else:
        most_common = int(np.bincount(predictions).argmax())
    
    print(f"\nPrediction Summary:")
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(predictions == class_idx)
        pct = count / len(predictions) * 100
        print(f"  {class_name}: {count} segments ({pct:.1f}%)")
    
    print(f"\nMost common prediction: {class_names[most_common]}")
    
    return (most_common, predictions, probabilities)


def get_feature_importance(
    model_path: str,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model_path: Path to saved model
        top_n: Number of top features to return
        
    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data.get('feature_names', None)
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    df = df.sort_values('importance', ascending=False).head(top_n).reset_index(drop=True)
    
    return df
