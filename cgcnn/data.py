from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader



def collate_pool(dataset_list):
    """Modified collate_pool function to handle extra features"""
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_extra_fea = []
    batch_cif_ids = []
    base_idx = 0
    
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, extra_fea), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_extra_fea.append(extra_fea)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
            torch.stack(batch_extra_fea, dim=0)),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids
        
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)







class CIFData(Dataset):
    def __init__(self, root_dir, feature_file, max_num_nbr=12, radius=8, dmin=0, step=0.2, 
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        

        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        self.extra_features = {}
        self.feature_normalizer = None
        self.feature_names = []

        assert os.path.exists(feature_file), 'Feature file does not exist!'
        self._load_extra_features(feature_file)
        self._compute_feature_normalization()

        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        """Return the size of dataset."""
        return len(self.id_prop_data)

    def _load_extra_features(self, feature_file):
        print("\nLoading extra features from file...")
        
        with open(feature_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            self.feature_names = header[1:]
            print(f"Detected {len(self.feature_names)} features: {', '.join(self.feature_names)}")
            
            self.default_values = {name: 0.0 for name in self.feature_names}
            
            for row in reader:
                try:
                    crystal_id = row[0]
                    features = [float(val) for val in row[1:]]
                    if len(features) != len(self.feature_names):
                        raise ValueError(f"Mismatch in feature count for {crystal_id}")
                    self.extra_features[crystal_id] = features
                    
                    if len(self.extra_features) <= 5:
                        feature_str = ", ".join([f"{self.feature_names[i]}: {val:.6f}" 
                                               for i, val in enumerate(features)])
                        print(f"Loaded features for {crystal_id}: {feature_str}")
                except (IndexError, ValueError) as e:
                    print(f"Error processing row in features file: {row}")
                    continue
                    
        print(f"Total number of crystals with extra features: {len(self.extra_features)}")

        print("\nFeature Statistics:")
        for i, name in enumerate(self.feature_names):
            values = [features[i] for features in self.extra_features.values()]
            print(f"{name}:")
            print(f"  Min: {min(values):.6f}")
            print(f"  Max: {max(values):.6f}")
            print(f"  Mean: {np.mean(values):.6f}")
            print(f"  Std: {np.std(values):.6f}")
        
    def _compute_feature_normalization(self):
        """Compute normalization statistics for all features."""
        if not self.extra_features:
            return

        feature_values = {name: [] for name in self.feature_names}

        for features in self.extra_features.values():
            for i, value in enumerate(features):
                feature_values[self.feature_names[i]].append(value)

        self.feature_normalizer = {}
        for name, values in feature_values.items():
            self.feature_normalizer[name] = {
                'mean': np.mean(values),
                'std': np.std(values) or 1.0
            }

    def _normalize_features(self, features):
        """Normalize all features using computed statistics."""
        if self.feature_normalizer is None:
            return torch.tensor(features, dtype=torch.float32)

        normalized_features = []
        for i, feature in enumerate(features):
            name = self.feature_names[i]
            if abs(feature) > 100:
                feature = np.clip(feature, -100, 100)

            norm_value = (feature - self.feature_normalizer[name]['mean']) / \
                    max(self.feature_normalizer[name]['std'], 1e-8)
                        
            norm_value = np.clip(norm_value, -5.0, 5.0)
            normalized_features.append(norm_value)
        
        return torch.tensor(normalized_features, dtype=torch.float32)

    
    
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        
        if cif_id in self.extra_features:
            features = self.extra_features[cif_id]
        else:
            warnings.warn(f"No extra features found for crystal {cif_id}. Using default values.")
            features = [self.default_values[name] for name in self.feature_names]
        
        extra_fea = self._normalize_features(features)
        
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                             for i in range(len(crystal))])        
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(f'{cif_id} has fewer than {self.max_num_nbr} neighbors')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                 [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                             [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                          nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                      nbr[:self.max_num_nbr])))
        
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(self.gdf.expand(np.array(nbr_fea)))
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])

        return (atom_fea, nbr_fea, nbr_fea_idx, extra_fea), target, cif_id      
