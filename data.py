"""
PyTorch data loading
"""
from typing import Optional

import torch
import numpy as np
import csv
import h5py
from torch.utils.data import Dataset
import os
import random

class Dataset3DThermal(Dataset):
    def __init__(self, file_name, R_range, group, feature_idx=None):
        self.file_name = file_name
        self.R_range = R_range
        self.group = group

        if feature_idx is None:
            feature_idx = slice(None)

        self.feature_idx = feature_idx
        self.features, self.kappa = self.load_data()
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.kappa[idx]

    def load_data(self):
        features_list = []
        kappa_list = []

        with h5py.File(self.file_name, "r") as F:
            feature_vectors = torch.tensor(F[f"{self.group}/feature_vector"][...], dtype=torch.float32)

            # Truncate feature vector
            feature_vectors = feature_vectors[..., self.feature_idx]
            
            # This is hacky, but we only want to use a subset of the data for validation
            if self.group == 'structures_val':
                feature_vectors = feature_vectors[:2000]

            num_samples = feature_vectors.shape[0]

            for R in self.R_range:
                R_column = torch.ones((num_samples, 1)) * R
                onebyR_column = torch.ones((num_samples, 1)) / R
                features_with_R = torch.hstack((feature_vectors, onebyR_column, R_column))
                features_list.append(features_with_R)

                if R < 1:  # For fractional R, the corresponding kappa is stored as contrast_{R_value}_inv
                    R_key = int(round(1 / R))
                    kappa_grp = torch.tensor(F[f"{self.group}/effective_conductivity/contrast_invR_{R_key}"][...], dtype=torch.float32)
                elif R > 1:
                    R_key = int(round(R))
                    kappa_grp = torch.tensor(F[f"{self.group}/effective_conductivity/contrast_R_{R_key}"][...], dtype=torch.float32)
                else:
                    kappa_grp = torch.ones((num_samples, 6))
                    kappa_grp[..., 3:] = 0.
                kappa_list.append(kappa_grp)

        # Concatenate all features and kappa arrays vertically
        all_features = torch.vstack(features_list)
        # Make the 0th feature -> 1 - 0th feature
        # 0th feature is the porosity (volume fraction of phase 0) - we want the volume fraction of phase 1
        all_features[:, 0] = 1.0 - all_features[:, 0]
        all_kappa = torch.vstack(kappa_list)
        # Scale components from dimension 2 onwards
        all_kappa[:, 3:] = all_kappa[:, 3:] / np.sqrt(2.0)

        # Convert to PyTorch tensors
        features_tensor = all_features
        kappa_tensor = all_kappa
        
        #Check all values are finite
        if not torch.all(torch.isfinite(features_tensor)):
            raise ValueError("Invalid features_tensor: contains non-finite values in group {}".format(self.group))
        if not torch.all(torch.isfinite(kappa_tensor)):
            raise ValueError("Invalid kappa_tensor: contains non-finite values in group {}".format(self.group))

        return features_tensor, kappa_tensor

class Dataset3DMechanical(Dataset):
    def __init__(self, 
                 csv_file_path, 
                 h5_file_path, 
                 group,
                 num_samples, 
                 feature_vector_name='feature_vector', 
                 random_seed=42,
                 device='cpu',
                 dtype=torch.float32,
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

        with open(self.csv_file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return [
                {
                    'dataset_index': int(row['dataset_index']),
                    'alpha': float(row['alpha']),
                    'beta': float(row['beta']),
                    'gamma': float(row['gamma']),
                    'hash': row['hash']
                }
                for row in reader
                if row['dataset_name'] == self.group.replace('structures_', '')
            ]

    def _load_data(self):
        if not os.path.exists(self.h5_file_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file_path}")

        with h5py.File(self.h5_file_path, 'r') as f:
            feature_vector_path = f"/{self.group}/{self.feature_vector_name}"
            if feature_vector_path not in f:
                raise KeyError(f"Feature vector dataset not found at {feature_vector_path}")

            feature_vectors = f[feature_vector_path]
            
            features_list = []
            C_list = []

            # First, just load and store all data in lists
            for entry in self.sampled_entries:
                i = entry['dataset_index']
                alpha, beta, gamma = entry['alpha'], entry['beta'], entry['gamma']
                hash_str = entry['hash']

                # Load and slice features
                feat = torch.tensor(feature_vectors[i, :], dtype=self.dtype)[self.feature_idx]

                # Append parameters
                appended = torch.tensor([1/alpha, 1/beta, 1/gamma, alpha, beta, gamma], dtype=self.dtype)
                final_feat = torch.cat([feat, appended])
                features_list.append(final_feat)

                # Load tangent (6x6)
                tangent_path = f"/{self.group}/dset_{i}/image/{hash_str}/load0/time_step0/homogenized_tangent"
                # if tangent_path not in f:
                #     raise KeyError(f"Homogenized tangent not found at {tangent_path}")

                C_np = f[tangent_path][...]
                # if C_np.shape != (6,6):
                #     raise ValueError(f"Expected 6x6 tangent, got {C_np.shape}")

                C_t = torch.tensor(C_np, dtype=self.dtype)
                C_list.append(C_t)

            # Now stack all features and tangents
            self.all_features = torch.stack(features_list, dim=0).to(self.device)   # (N, selected_features+6)
            self.all_features[:, 0] = 1.0 - self.all_features[:, 0]                 # Invert the volume fraction    
            C_all_6x6 = torch.stack(C_list, dim=0).to(self.device)                  # (N, 6, 6)

            # Convert all tangents at once
            self.all_C = C6x6_to_C21(C_all_6x6)  # (N,21)

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

def Piso1() -> torch.Tensor:
    """Returns the first isotropic projector in Mandel notation."""
    P = torch.zeros((6, 6), dtype=torch.float32)
    P[:3, :3] = 1. / 3.
    return P

def Piso2() -> torch.Tensor:
    """Returns the second isotropic projector in Mandel notation."""
    P = torch.eye(6, dtype=torch.float32)
    P = P - Piso1()
    return P

def Ciso(K: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Returns an isotropic stiffness tensor in Mandel notation."""
    P1 = Piso1().to(K.device)
    I6 = torch.eye(6, dtype=torch.float32).to(K.device)
    
    if K.ndimension() == 1 and G.ndimension() == 1:
        return (3. * K - 2. * G)[:, None, None] * P1[None, :, :] + 2. * G[:, None, None] * I6[None, :, :]
    else:
        return (3. * K - 2. * G) * P1 + 2. * G * I6