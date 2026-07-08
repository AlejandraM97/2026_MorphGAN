import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize


class NoduleDataset(Dataset):
    def __init__(self, hist_df, npy_dir):
        """
        Args:
            hist_df (pd.DataFrame): DataFrame with 'nodule_id' and 'malignancy' columns.
            npy_dir (str): Directory containing .npy files for nodules.
        """
        # Filter conditions
        hist_df = hist_df.loc[
            (hist_df['malignancy'] > 0) &
            (hist_df['method'] != 0) &
            (hist_df['malignancy'] != 3.0)
        ]
        hist_df = hist_df[~hist_df['nodule_id'].str.startswith("LIDC-IDRI-0332")]
        hist_df.loc[:, 'malignancy'] = hist_df['malignancy'].map(lambda x: 0 if x == 1 else 1)

        self.npy_dir = npy_dir
        self.nodule_ids = []
        self.labels = []
        self.missing_files = []

        for nodule_id, label in zip(hist_df['nodule_id'], hist_df['malignancy']):
            file_path = os.path.join(npy_dir, f"{nodule_id}.npy")
            matching_files = glob.glob(file_path)

            if not matching_files:
                self.missing_files.append(nodule_id)
                continue
            if len(matching_files) > 1:
                raise ValueError(f"Multiple files found for nodule ID: {nodule_id} - {matching_files}")

            self.nodule_ids.append(matching_files[0])
            self.labels.append(label)

        if self.missing_files:
            print(f"Warning: {len(self.missing_files)} missing .npy files")

        if not self.nodule_ids:
            raise ValueError("No valid samples found for NoduleDataset.")

    def __len__(self):
        return len(self.nodule_ids)

    def __getitem__(self, idx):
        file_path = self.nodule_ids[idx]
        label = self.labels[idx]

        npy_data = np.load(file_path)
        npy_data_resized = resize(npy_data, (32, 32), mode='wrap', anti_aliasing=False)

        npy_min, npy_max = npy_data_resized.min(), npy_data_resized.max()
        if npy_max > npy_min:
            npy_data_resized = (npy_data_resized - npy_min) / (npy_max - npy_min)
        else:
            npy_data_resized = npy_data_resized - npy_min

        npy_data_resized = np.expand_dims(npy_data_resized, axis=0)  # Shape: [1, 32, 32]
        return torch.tensor(npy_data_resized).float(), torch.tensor(label).float()


def create_hist_loader(csv_path, npy_dir, batch_size=16, shuffle=False):
    """
    Creates a DataLoader from the histogram metadata and .npy files.
    """
    hist_df = pd.read_csv(csv_path)
    dataset = NoduleDataset(hist_df, npy_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
