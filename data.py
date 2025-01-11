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
        device='cpu', 
        dtype=torch.float64, 
        feature_idx=None,
        max_samples=None
    ):
        """
        A PyTorch Dataset for 3D thermal microstructure data.

        Args:
            h5_file_path (str): Path to the HDF5 file with data.
            group (str): One of 'structures_train', 'structures_val', 'structures_test'.
            R_range (iterable): Range of contrast values R to append to each sample.
            feature_vector_name (str): Name of the feature vector dataset in HDF5.
            device (str or torch.device): Device for final tensors ('cpu', 'cuda', etc.).
            dtype (torch.dtype): Data type of final tensors (e.g., torch.float64).
            feature_idx (None or sequence): Indices of features to keep. If None, keep all.
            max_samples (int or None): If set, limit the dataset to the first 'max_samples'.

        Raises:
            FileNotFoundError: If the HDF5 file is missing.
            KeyError: If feature_vector_name is not found in HDF5.
            ValueError: If the required kappa dataset for R is missing, or non-finite values occur.
        """
        self.h5_file_path = h5_file_path
        self.group = group
        self.R_range = R_range
        self.feature_vector_name = feature_vector_name
        self.device = device
        self.dtype = dtype
        self.feature_idx = slice(None) if (feature_idx is None) else feature_idx
        self.max_samples = max_samples

        self.features, self.kappa = self._load_data()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.kappa[idx]

    def _load_data(self):
        """Load features and kappa from HDF5, expand them for each R, and return torch Tensors."""
        if not os.path.exists(self.h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        with h5py.File(self.h5_file_path, "r") as f:
            # Load base feature vectors
            feature_path = f"{self.group}/{self.feature_vector_name}"
            if feature_path not in f:
                raise KeyError(f"Feature vector dataset not found at {feature_path}")
            
            base_features = f[feature_path][...]
            # Truncate feature vector
            base_features = base_features[..., self.feature_idx]
            
            # Optionally limit the dataset size (max_samples)
            if self.max_samples is not None:
                base_features = base_features[: self.max_samples]

            num_samples = base_features.shape[0]
            feature_dim = base_features.shape[1] + 2
            total_count = num_samples * len(self.R_range)

            all_features_np = np.empty((total_count, feature_dim), dtype=np.float64)
            all_kappa_np = np.empty((total_count, 6), dtype=np.float64)

            # For each R, copy base_features + columns [1/R, R], load the matching kappa
            for i, R in enumerate(self.R_range):
                start = i * num_samples
                end = (i + 1) * num_samples

                # Build final features for this R
                all_features_np[start:end, :-2] = base_features
                all_features_np[start:end, -2] = 1.0 / R
                all_features_np[start:end, -1] = R

                # Load kappa
                try:
                    if R < 1:
                        R_key = int(round(1.0 / R))
                        kappa_data = f[f"{self.group}/effective_conductivity/contrast_invR_{R_key}"][...]
                    elif R > 1:
                        R_key = int(round(R))
                        kappa_data = f[f"{self.group}/effective_conductivity/contrast_R_{R_key}"][...]
                    else:
                        # R == 1
                        kappa_data = np.ones((num_samples, 6), dtype=np.float64)
                        kappa_data[:, 3:] = 0.0
                except KeyError:
                    raise ValueError(
                        f"No valid kappa dataset found for R={R} in group '{self.group}'. "
                        f"Looked for 'contrast_invR_{R_key}' or 'contrast_R_{R_key}'"
                    )

                # Possibly limit kappa_data to max_samples if it is large
                if self.max_samples is not None:
                    kappa_data = kappa_data[:self.max_samples]

                if kappa_data.shape != (num_samples, 6):
                    raise ValueError(
                        f"Expected kappa shape ({num_samples}, 6) for R={R}, got {kappa_data.shape}"
                    )

                all_kappa_np[start:end] = kappa_data

        # Invert volume fraction of phase 0 to get volume fraction of phase 1
        all_features_np[:, 0] = 1.0 - all_features_np[:, 0]

        # Scale kappa columns 3..5 by 1/sqrt(2)
        all_kappa_np[:, 3:] /= np.sqrt(2.0)        

        features_t = torch.from_numpy(all_features_np).to(dtype=self.dtype, device=self.device)
        kappa_t = torch.from_numpy(all_kappa_np).to(dtype=self.dtype, device=self.device)

        return features_t, kappa_t

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

        The homogenized tangent is a 6x6 symmetric matrix. We extract its lower triangle 
        in the order:
        (0,0), (1,1), (2,2), (3,3), (4,4), (5,5),
        (1,0),
        (2,0), (2,1),
        (3,0), (3,1), (3,2),
        (4,0), (4,1), (4,2), (4,3),
        (5,0), (5,1), (5,2), (5,3), (5,4)
        
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
        self.all_C = pack_sym(C_all_6x6, dim=6)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.all_features[idx], self.all_C[idx]

















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



# Functions for converting between symmetric matrix representations

def get_sym_indices(dim):
    diag_idx = (torch.arange(dim), torch.arange(dim))    
    row, col = torch.tril_indices(dim, dim, -1)
    dof_idx = (torch.cat([diag_idx[0], row]), torch.cat([diag_idx[1], col]))
    return dof_idx

def pack_sym(symmetric_matrix, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(symmetric_matrix.device) for idx in dof_idx)
    return symmetric_matrix[(..., *dof_idx) if symmetric_matrix.dim() == 3 else dof_idx]

def unpack_sym(packed_values, dim, dof_idx=None):
    if dof_idx is None:
        dof_idx = get_sym_indices(dim)
    dof_idx = tuple(idx.to(packed_values.device) for idx in dof_idx)
    matrix = torch.zeros((*packed_values.shape[:-1], dim, dim), dtype=packed_values.dtype, device=packed_values.device)
    if packed_values.dim() == 2:
        matrix[:, dof_idx[0], dof_idx[1]] = packed_values
        return matrix + matrix.transpose(1, 2) - torch.diag_embed(torch.diagonal(matrix, dim1=1, dim2=2))
    matrix[dof_idx] = packed_values
    return matrix + matrix.T - torch.diag(torch.diag(matrix))