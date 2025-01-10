"""
PyTorch data loading
"""

import torch
import numpy as np
import csv
import h5py
from torch.utils.data import Dataset
import os
import random

class Dataset3DThermal(Dataset):
    def __init__(
        self, 
        h5_file_path: str, 
        group: str, 
        R_range, 
        feature_vector_name: str = 'feature_vector', 
        device = 'cpu', 
        dtype = torch.float64, 
        feature_idx = None
    ):
        """
        A PyTorch Dataset for 3D thermal microstructure data.

        Args:
            h5_file_path (str): Path to the HDF5 file with data.
            group (str): One of 'structures_train', 'structures_val', or 'structures_test'.
            R_range (iterable): Range of contrast values R to append to each sample.
            feature_vector_name (str): Name of the feature vector dataset in the HDF5.
            device (str or torch.device): Device to place the final tensors on.
            dtype (torch.dtype): Data type for the final tensors.
            feature_idx (None or sequence): Indices of features to keep. If None, keep all.
        """
        self.h5_file_path = h5_file_path
        self.group = group
        self.R_range = R_range
        self.feature_vector_name = feature_vector_name
        self.device = device
        self.dtype = dtype

        if feature_idx is None:
            feature_idx = slice(None)
        self.feature_idx = feature_idx

        self.features, self.kappa = self.load_data()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.kappa[idx]

    def load_data(self):
        """Loads data from HDF5, returns (features, kappa) as torch tensors."""
        if not os.path.exists(self.h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        with h5py.File(self.h5_file_path, "r") as f:
            # Load and truncate feature vectors
            features = f[f"{self.group}/{self.feature_vector_name}"][...][:, self.feature_idx]
            
            # !!!!! This is a hacky fix to reduce the number of samples for validation and test sets !!!!!!
            if self.group in ['structures_val', 'structures_test']:
                features = features[:2000]
            
            num_samples = len(features)
            feature_dim = features.shape[1] + 2  # +2 for R and 1/R columns
            
            # Pre-allocate arrays
            all_features = np.empty((num_samples * len(self.R_range), feature_dim), dtype=np.float64)
            all_kappa = np.empty((num_samples * len(self.R_range), 6), dtype=np.float64)
            
            # Fill arrays
            for i, R in enumerate(self.R_range):
                idx = slice(i*num_samples, (i+1)*num_samples)
                
                # Set features with R columns
                all_features[idx] = np.column_stack([features, np.full(num_samples, 1/R), np.full(num_samples, R)])
                
                # Set kappa values
                if R < 1:
                    R_key = int(round(1/R))
                    kappa = f[f"{self.group}/effective_conductivity/contrast_invR_{R_key}"][...]
                elif R > 1:
                    R_key = int(round(R))
                    kappa = f[f"{self.group}/effective_conductivity/contrast_R_{R_key}"][...]
                else:
                    kappa = np.ones((num_samples, 6), dtype=np.float64)
                    kappa[:, 3:] = 0
                    
                all_kappa[idx] = kappa

        all_features[:, 0] = 1.0 - all_features[:, 0]  # Invert volume fraction of phase 0 to get phase 1
        all_kappa[:, 3:] /= np.sqrt(2.0)  # Scale off-diagonal terms

        # Convert to torch tensors
        features_tensor = torch.from_numpy(all_features).to(device=self.device, dtype=self.dtype)
        kappa_tensor = torch.from_numpy(all_kappa).to(device=self.device, dtype=self.dtype)

        if not (torch.isfinite(features_tensor).all() and torch.isfinite(kappa_tensor).all()):
            raise ValueError(f"Non-finite values found in tensors for group {self.group}")

        return features_tensor, kappa_tensor

class Dataset3DMechanical(Dataset):
    def __init__(self, 
                 csv_file_path: str, 
                 h5_file_path: str, 
                 group: str,
                 num_samples: int, 
                 feature_vector_name='feature_vector', 
                 random_seed=42,
                 device='cpu',
                 dtype=torch.float64,
                 feature_idx=None):
        """
        A PyTorch Dataset for 3D mechanical microstructure homogenization data.
        
        The final feature vector is formed by:
        [selected_original_features, 1/alpha, 1/beta, 1/gamma, alpha, beta, gamma]

        The homogenized tangent is a 6x6 symmetric matrix. We extract its upper triangle 
        in the order:
        (0,0), (1,1), (2,2), (3,3), (4,4), (5,5),
        (0,1), (0,2), (0,3), (0,4), (0,5),
        (1,2), (1,3), (1,4), (1,5),
        (2,3), (2,4), (2,5),
        (3,4), (3,5),
        (4,5)
        
        Args:
            csv_file_path (str): Path to the CSV file with metadata.
            h5_file_path (str): Path to the HDF5 file with data.
            group (str): 'structures_train', 'structures_val', or 'structures_test'.
            num_samples (int): Number of samples to randomly select.
            feature_vector_name (str): Name of the feature vector dataset in the HDF5.
            random_seed (int): Seed for reproducibility.
            device (str or torch.device): Device for final tensors.
            dtype (torch.dtype): Data type of final tensors.
            feature_idx (None or sequence): Indices of features to keep. If None, keep all.
        """
        # Validate group
        if group not in ['structures_train', 'structures_val', 'structures_test']:
            raise ValueError("group must be one of ['structures_train', 'structures_val', 'structures_test']")

        self.csv_file_path = csv_file_path
        self.h5_file_path = h5_file_path
        self.group = group
        self.feature_vector_name = feature_vector_name
        self.device = device
        self.dtype = dtype

        if feature_idx is None:
            feature_idx = slice(None)
        self.feature_idx = feature_idx

        entries = self._load_csv_and_filter()

        if num_samples > len(entries):
            raise ValueError(f"Requested {num_samples} samples, only {len(entries)} available for {group}.")

        random.seed(random_seed)
        self.sampled_entries = random.sample(entries, num_samples)
        self.num_samples = num_samples

        self._load_data()

    def _load_csv_and_filter(self):
        """Load and filter the CSV data for the specified group in a compact manner."""
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        target_name = self.group.replace('structures_', '')
        required_fields = ['dataset_index', 'alpha', 'beta', 'gamma', 'hash']
        entries = []
        with open(self.csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['dataset_name'] == target_name:
                    entries.append({
                        field: (int if field == 'dataset_index' else float if field != 'hash' else str)(row[field])
                        for field in required_fields
                    })
        return entries

    def _load_data(self):
        if not os.path.exists(self.h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        with h5py.File(self.h5_file_path, 'r') as f:
            feature_vector_path = f"/{self.group}/{self.feature_vector_name}"
            if feature_vector_path not in f:
                raise KeyError(f"Feature vector dataset not found at {feature_vector_path}")
            
            feature_vectors = f[feature_vector_path]
            # Truncate feature vector
            feature_vectors = feature_vectors[..., self.feature_idx]

            n_samples = len(self.sampled_entries)
            n_features = feature_vectors.shape[1] + 6  # original features + 6 additional
            features_np = np.empty((n_samples, n_features), dtype=np.float64)
            tangents_np = np.empty((n_samples, 6, 6), dtype=np.float64)

            for idx, entry in enumerate(self.sampled_entries):
                i = entry['dataset_index']
                alpha, beta, gamma = entry['alpha'], entry['beta'], entry['gamma']
                hash_str = entry['hash']
                
                features_np[idx] = np.concatenate([
                    feature_vectors[i,:],
                    [1/alpha, 1/beta, 1/gamma, alpha, beta, gamma]
                ])
                                
                tangent_path = f"/{self.group}/dset_{i}/image/{hash_str}/load0/time_step0/homogenized_tangent"
                tangents_np[idx] = f[tangent_path][...]

            # Invert volume fraction of phase 0 to get volume fraction of phase 1
            features_np[:, 0] = 1.0 - features_np[:, 0]

        # Convert to torch tensors
        self.all_features = torch.from_numpy(features_np).to(dtype=self.dtype, device=self.device)
        C_all_6x6 = torch.from_numpy(tangents_np).to(dtype=self.dtype, device=self.device)
        self.all_C = C6x6_to_C21(C_all_6x6)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.all_features[idx], self.all_C[idx]

















def C6x6_to_C21(C_6x6: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 6x6 symmetric matrices to their 21-element upper-triangular representation.
    
    Args:
        C_6x6 (torch.Tensor): Shape (N, 6, 6), symmetric matrices.

    Returns:
        torch.Tensor: Shape (N, 21), flattened upper-triangular elements.
    """
    indices = torch.tensor([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        [3, 4], [3, 5],
        [4, 5]
    ], dtype=torch.long)
    
    return C_6x6[:, indices[:, 0], indices[:, 1]]  # (N, 21)

def C21_to_C6x6(C_21: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 21-element vectors to their full 6x6 symmetric matrix form.
    
    Args:
        C_21 (torch.Tensor): Shape (N, 21), upper-triangular elements.

    Returns:
        torch.Tensor: Shape (N, 6, 6) symmetric matrices.
    """
    indices = torch.tensor([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 3], [2, 4], [2, 5],
        [3, 4], [3, 5],
        [4, 5]
    ], dtype=torch.long)
    
    N = C_21.shape[0]
    C_6x6 = torch.zeros((N, 6, 6), dtype=C_21.dtype, device=C_21.device)
    C_6x6[:, indices[:, 0], indices[:, 1]] = C_21
    C_6x6[:, indices[:, 1], indices[:, 0]] = C_21

    return C_6x6

def Piso1(dtype=torch.float64) -> torch.Tensor:
    """Returns the first isotropic projector in Mandel notation."""
    P = torch.zeros((6, 6), dtype=dtype)
    P[:3, :3] = 1. / 3.
    return P

def Piso2(dtype=torch.float64) -> torch.Tensor:
    """Returns the second isotropic projector in Mandel notation."""
    P = torch.eye(6, dtype=dtype)
    P = P - Piso1(dtype=dtype)
    return P

def Ciso(K: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Returns an isotropic stiffness tensor in Mandel notation."""
    P1 = Piso1(dtype=K.dtype).to(K.device)
    I6 = torch.eye(6, dtype=K.dtype).to(K.device)
    
    if K.ndimension() == 1 and G.ndimension() == 1:
        return (3. * K - 2. * G)[:, None, None] * P1[None, :, :] + 2. * G[:, None, None] * I6[None, :, :]
    else:
        return (3. * K - 2. * G) * P1 + 2. * G * I6