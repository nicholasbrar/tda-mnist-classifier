# src/features.py
import numpy as np
import gudhi
from tqdm import tqdm
from joblib import Parallel, delayed
from preprocessing import binarize_digit

def get_all_features(X):
    """
    Extracts topological features for all images in the dataset.

    Args:
        X (np.ndarray): Array of shape (n_samples, 784) with flattened MNIST images.

    Returns:
        np.ndarray: Array of shape (n_samples, n_features) with topological features.
    """
    features = []
    for img in tqdm(X, desc="Extracting features"):
        binary_img = binarize_digit(img)
        feats = extract_tda_features(binary_img)
        features.append(feats)
    return np.array(features)

def extract_tda_features(binary_img, max_dist=1.5):
    """
    Extracts topological data analysis (TDA) features from a binarized 28x28 MNIST image.

    Args:
        binary_img (np.ndarray): 28x28 binary image (pixels 0 or 255).
        max_dist (float, optional): Maximum edge length for Rips complex (default 1.5).

    Returns:
        np.ndarray: Array of TDA features.
    """
    points = np.argwhere(binary_img > 0)
    
    # Initialize comprehensive feature dictionary
    features = {
        'h0_count': 0, 'h0_total_persistence': 0, 'h0_max': 0,
        'h1_count': 0, 'h1_total_persistence': 0, 'h1_max': 0,
        'h1_birth_mean': 0, 'h1_death_mean': 0,
        'h0_entropy': 0, 'h1_entropy': 0,
        'num_components': 0,
        'num_loops': 0,
        'h0_persistence_ratio': 0
    }
    
    if len(points) < 2:
        return np.array(list(features.values()))
    
    # Normalize coordinates to [0,1] range
    points_norm = points / 27.0
    
    # Compute persistence
    rc = gudhi.RipsComplex(points=points_norm, max_edge_length=max_dist)
    st = rc.create_simplex_tree(max_dimension=2)
    persistence = st.persistence()
    
    # Process persistence features
    h0 = [(b,d) for dim, (b,d) in persistence if dim == 0]
    h1 = [(b,d) for dim, (b,d) in persistence if dim == 1]
    
    # H0 features (components)
    if h0:
        pers = [d-b for b,d in h0 if not np.isinf(d)]
        features['h0_count'] = len(pers)
        features['h0_total_persistence'] = sum(pers)
        features['h0_max'] = max(pers) if pers else 0
        features['num_components'] = len([1 for b,d in h0 if np.isinf(d)])
        
        # Persistence entropy
        total_pers = sum(pers)
        if total_pers > 0:
            features['h0_entropy'] = -sum((p/total_pers)*np.log(p/total_pers+1e-10) for p in pers)
    
    # H1 features (holes)
    if h1:
        pers = [d-b for b,d in h1]
        features['h1_count'] = len(pers)
        features['h1_total_persistence'] = sum(pers)
        features['h1_max'] = max(pers) if pers else 0
        features['h1_birth_mean'] = np.mean([b for b,d in h1])
        features['h1_death_mean'] = np.mean([d for b,d in h1])
        features['num_loops'] = len(h1)
        
        # Persistence entropy
        total_pers = sum(pers)
        if total_pers > 0:
            features['h1_entropy'] = -sum((p/total_pers)*np.log(p/total_pers+1e-10) for p in pers)
    
    if features['num_components'] > 0:
        features['h0_persistence_ratio'] = features['h0_total_persistence'] / features['num_components']
    return np.array(list(features.values()))