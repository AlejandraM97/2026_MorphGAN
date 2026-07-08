import os
import json
import pandas as pd
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
from skimage.transform import resize
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
from sklearn.model_selection import train_test_split
import random

from config.options import Options
from config import registry
opt = Options().parse()
registry.opt = opt


def create_dataset(version: str):
    """
    Returns the appropriate dataset class based on the specified version.

    Args:
        version (str): Version identifier for the dataset. Must be one of:
            - 'v3': Uses CombinedDatasetV3 for folder-based data loading.
            - 'v4': Uses NLSTDatasetV1 for a training considering benign and malign nodules on an external NLST dataset.
            - 'v5': Uses NLSTDatasetBenignV1 for training only LUNA25 benign nodules on an external NLST dataset.

    Returns:
        Dataset class: The dataset class corresponding to the version.

    Raises:
        ValueError: If the specified version is invalid.
    """
    datasets = {
        'v3': CombinedDatasetV3,
        'v4': NLSTDatasetV1,
        'v5': NLSTDatasetBenignV1,
        'v6': PathName_MC_DatasetV4,
        'v7': LNDbDatasetBenignV1,
        'v8': NLSTDatasetBenign_ZeroFull,
        'v9': LIDC_malignant_dataset
    }

    if version not in datasets:
        available_versions = ', '.join(datasets.keys())
        raise ValueError(f"Invalid dataset version '{version}'. Available versions are: {available_versions}")

    return datasets[version]

# Configure logging
log_path = './results/logs'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'data_loading.log')),
        logging.StreamHandler()
    ]
)

# -----------------------
# 1. Helper Functions
# -----------------------

def load_npy_file(file_path: str) -> np.ndarray:
    """Loads an .npy file and returns a NumPy array."""
    try:
        return np.load(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Failed to load file {file_path}: {e}")
        return None

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Applies Min-Max normalization to an image."""
    min_val, max_val = image.min(), image.max()
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image - min_val

def resize_image(image: np.ndarray, shape=(1, 32, 32)) -> np.ndarray:
    """Resizes an image to the specified shape."""
    return resize(image, shape, mode='wrap', anti_aliasing=False)

def prepare_data(file_path: str) -> torch.Tensor:
    """Loads, normalizes, and converts an image to a tensor."""
    npy_data = load_npy_file(file_path)
    if npy_data is None:
        return None
    npy_data = normalize_image(npy_data)
    npy_data = resize_image(np.expand_dims(npy_data, axis=0))
    return torch.from_numpy(npy_data).float()

# -----------------------
# 2. CustomDatasetV3 Class
# -----------------------

class CustomDatasetV3(Dataset):
    """
    Dataset class for version 3: Loads data from folders.

    Args:
        folder_path (str): Path to the base folder containing 'train' and 'test' folders.
        dataset_type (str): 'train' or 'test'.
    """
    def __init__(self, folder_path: str, dataset_type: str = 'train') -> None:
        self.folder_path = os.path.join(folder_path, dataset_type)
        self.dataset_type = dataset_type
        self.file_paths = []
        self.labels = []

        self._load_data_from_folders()
        self._print_nodule_counts()

    def _load_data_from_folders(self) -> None:
        """Loads file paths and labels from folder structure."""
        benign_folder = os.path.join(self.folder_path, 'benign')
        malignant_folder = os.path.join(self.folder_path, 'malignant')

        if os.path.exists(benign_folder):
            for file_name in os.listdir(benign_folder):
                if file_name.endswith('.npy'):
                    self.file_paths.append(os.path.join(benign_folder, file_name))
                    self.labels.append(0)  # Benign label

        if os.path.exists(malignant_folder):
            for file_name in os.listdir(malignant_folder):
                if file_name.endswith('.npy'):
                    self.file_paths.append(os.path.join(malignant_folder, file_name))
                    self.labels.append(1)  # Malignant label

        logging.info(f"Loaded {len(self.file_paths)} samples from {self.dataset_type} folder.")

    def _print_nodule_counts(self) -> None:
        """Prints the count of benign and malignant nodules."""
        benign_count = self.labels.count(0)
        malignant_count = self.labels.count(1)
        logging.info(f"{self.dataset_type.capitalize()} Dataset: {benign_count} benign, {malignant_count} malignant nodules.")
    
    def _get_file_paths_(self) -> list:
        """Returns the list of file paths."""
        return self.file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        tensor = prepare_data(file_path)
        if tensor is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        return tensor, torch.tensor(label).float()

# -----------------------
# 3. CombinedDatasetV3 Class
# -----------------------

class CombinedDatasetV3(Dataset):
    """
    Combined Dataset class for managing train and test datasets.

    Args:
        base_folder (str): Path to the base directory containing 'train' and 'test' folders.
    """
    def __init__(self, base_folder: str) -> None:
        self.datasets = {
            'train': CustomDatasetV3(base_folder, dataset_type='train'),
            'test': CustomDatasetV3(base_folder, dataset_type='test')
        }

    def __getitem__(self, dataset_type: str) -> Dataset:
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return self.datasets[dataset_type]

# -----------------------
# 4. create_dataset Function
# -----------------------

def create_dataset(version: str):
    """
    Creates and returns the appropriate dataset class based on the specified version.

    Args:
        version (str): Version identifier for the dataset.

    Returns:
        Dataset: A dataset class instance based on the version.

    Raises:
        ValueError: If an invalid dataset version is specified.
        FileNotFoundError: If the data path does not exist.
    """
    if not os.path.exists(opt.folder_path):
        raise FileNotFoundError(f"Data path not found: {opt.folder_path}")

    datasets = {
        'v3': lambda: CombinedDatasetV3(opt.folder_path),
        'v4': lambda: NLSTDatasetV1(opt.folder_path),
        'v5': lambda: NLSTDatasetBenignV1(opt.folder_path),
        'v6': lambda: PathName_MC_DatasetV4(opt.folder_path),
        'v7': lambda: LNDbDatasetBenignV1(opt.folder_path),
        'v8': lambda: NLSTDatasetBenign_ZeroFull(opt.folder_path),
        'v9': lambda: LIDC_malignant_dataset(opt.folder_path)
    }

    if version not in datasets:
        available_versions = ', '.join(datasets.keys())
        raise ValueError(f"Invalid dataset version '{version}'. Available versions: {available_versions}")

    logging.info(f"Creating dataset for version: {version}")
    return datasets[version]()

# -----------------------
# 5. CustomNLSTDatasetV1 Class
# -----------------------

class CustomNLSTDatasetV1(Dataset):
    """
    Dataset class for the NLST dataset version 1.

    Args:
        data_path (str): Path to the NLST dataset folder.
        fold (int): Fold number for cross-validation.
    """
    def __init__(self, folder_path: str, dataset_type: str = 'train') -> None:
        self.folder_path = os.path.join(folder_path, dataset_type)
        self.dataset_type = dataset_type
        self.binary_labels = []
        self.file_paths = []

        # Load the dataframe from folders
        self.df = self._load_dataframe_from_folders()
        self._load_data_from_folders()
        self._print_nodule_counts()

    def _load_dataframe_from_folders(self) -> pd.DataFrame:
        malignant_nodules = '/data/Datasets/Lungs/LUNA25-NLST/2D_patches_malign/'
        benign_nodules = '/data/Datasets/Lungs/LUNA25-NLST/2D_patches_benign/'
        malignant_folders = np.array([os.path.join(malignant_nodules, folder) for folder in np.sort(os.listdir(malignant_nodules))])[:91]
        benign_folders = np.array([os.path.join(benign_nodules, folder) for folder in np.sort(os.listdir(benign_nodules))])[:91]
        malignant_label = np.array([1 for _ in range(len(malignant_folders))])
        benign_label = np.array([0 for _ in range(len(benign_folders))])

        #Dataframe with patient_id, malignant/benign label and path
        malignant_df = pd.DataFrame(malignant_folders, columns=['folder'])
        malignant_df['patient_id'] = np.sort(os.listdir(malignant_nodules))[:91]
        malignant_df['patient_id'] = malignant_df['patient_id'].astype(int)
        malignant_df['malignancy'] = malignant_label


        benign_df = pd.DataFrame(benign_folders, columns=['folder'])
        benign_df['patient_id'] = np.sort(os.listdir(benign_nodules))[:91]
        benign_df['patient_id'] = benign_df['patient_id'].astype(int)
        benign_df['malignancy'] = benign_label

        nlst_df = pd.concat([malignant_df, benign_df], ignore_index=True)

        chile_dataset = '/data/arumota/Nodules_Ohif/patch_chile/'
        path_folders = [os.path.join(chile_dataset, patient) for patient in sorted(os.listdir(chile_dataset))]

        chile_df = pd.read_csv('/data/arumota/PhD/Automation_bias/chile_malignancy.csv', sep=';')
        chile_df.rename(columns={'class': 'malignancy', 'patient': 'patient_id'}, inplace=True)
        chile_df['patient_id'] = chile_df['patient_id'].astype(int)

        #Add the paths on the dataframe
        chile_df['folder'] = path_folders

        df_nlst_chile_merge = nlst_df.merge(chile_df, on=['patient_id'], how='outer')
        df_nlst_chile_merge.rename(columns={'folder_y': 'folder', 'malignancy_y': 'malignancy'}, inplace=True)

        # Merge folder_x and folder if folder_x is NaN
        df_nlst_chile_merge['folder'] = df_nlst_chile_merge['folder'].combine_first(df_nlst_chile_merge['folder_x'])
        df_nlst_chile_merge['malignancy'] = df_nlst_chile_merge['malignancy'].combine_first(df_nlst_chile_merge['malignancy_x'])
        del df_nlst_chile_merge['folder_x']; del df_nlst_chile_merge['malignancy_x']

        df = pd.concat([df_nlst_chile_merge, nlst_df], ignore_index=True)
        df = df.drop_duplicates(subset=['patient_id'], keep='first')
        return df

    def _load_data_from_folders(self) -> None:
        """Loads file paths and binary_labels according to the dataframe."""
        # Make a training and testing sets with NLST and Chile dataset

        X_train, X_test, _, _ = train_test_split(self.df['patient_id'], self.df['malignancy'], 
                                                 test_size=0.99, random_state=42)

        if self.dataset_type == 'train':
            self.df = self.df[self.df['patient_id'].isin(X_train)]
        elif self.dataset_type == 'test':
            self.df = self.df[self.df['patient_id'].isin(X_test)]

        chile_dataset = "/data/arumota/Nodules_Ohif/patch_chile"
        for index, patient in enumerate(self.df['patient_id']):
            patient_folder = self.df.loc[self.df['patient_id'] == patient, 'folder'].values[0]
            patient_malignancy = self.df.loc[self.df['patient_id'] == patient, 'malignancy'].values[0]

            self.binary_labels.append(patient_malignancy)
            if patient_folder.startswith(chile_dataset):
                patient_folder = os.path.join(patient_folder, 'N_0')
            else:
                list_folders = os.listdir(patient_folder)
                patient_folder = os.path.join(patient_folder, list_folders[0])

            image_folder = sorted(os.listdir(patient_folder))
            middle_value    = len(image_folder)//2

            tensor_image = image_folder[middle_value]
            self.file_paths.append(os.path.join(patient_folder, tensor_image))
            

        logging.info(f"Loaded NLST samples from {self.dataset_type} folder.")

    def _print_nodule_counts(self) -> None:
        """Prints the count of benign and malignant nodules."""
        benign_count = self.binary_labels.count(0)
        malignant_count = self.binary_labels.count(1)
        logging.info(f"{self.dataset_type.capitalize()} Dataset: {benign_count} benign, {malignant_count} malignant nodules.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        binarys_label = self.binary_labels[idx]
        tensor = prepare_data(file_path)
        if tensor is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        return tensor, torch.tensor(binarys_label).float() # torch.tensor(multi_class_label).float() #file_path,



# -----------------------
# 6. NLSTDatasetV1 Class
# -----------------------

class NLSTDatasetV1(Dataset):
    """
    Combined Dataset class for managing train and test datasets.

    Args:
        base_folder (str): Path to the base directory containing 'train' and 'test' folders.
    """
    def __init__(self, base_folder: str) -> None:
        self.datasets = {
            'train': CustomNLSTDatasetV1(base_folder, dataset_type='train'),
            'test': CustomNLSTDatasetV1(base_folder, dataset_type='test')
        }

    def __getitem__(self, dataset_type: str) -> Dataset:
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return self.datasets[dataset_type]


# -----------------------
# 7. CustomPathName_MC_DatasetV4 Class
# -----------------------

class CustomPathName_MC_DatasetV4(Dataset):
    """
    Dataset class for version 4: Loads data from folders and compare the multiple class.

    Args:
        folder_path (str): Path to the base folder containing 'train' and 'test' folders.
        dataset_type (str): 'train' or 'test'.
    """
    def __init__(self, folder_path: str, dataset_type: str = 'train') -> None:
        self.folder_path = os.path.join(folder_path, dataset_type)
        self.dataset_type = dataset_type
        self.file_paths = []
        self.binary_labels = []
        self.multi_class_labels = []

        self._load_data_from_folders()
        self._print_nodule_counts()

    def _load_data_from_folders(self) -> None:
        """Loads file paths and binary_labels from folder structure."""
        benign_folder = os.path.join(self.folder_path, 'benign')
        malignant_folder = os.path.join(self.folder_path, 'malignant')
        df = pd.read_csv('/data/arumota/PhD/Pasantia/GANLung_Ale_paris/4_radiologists.csv')
        df['patient_id'] = df['patient_id'].astype(str)
        df['nodule_number'] = df['nodule_number'].astype(str)

        if os.path.exists(benign_folder):
            for file_name in os.listdir(benign_folder):
                if file_name.endswith('.npy'):
                    patient_id_benign = str(file_name.split('_')[0])  # Extract case name 
                    nodule_number_benign = str(file_name.split('_')[1])  # Extract nodule number
                    
                    filtered_benign = df[
                        (df['patient_id'] == patient_id_benign) &
                        (df['nodule_number'] == nodule_number_benign)
                    ]
                    
                    benign_value = filtered_benign.malignancy.values[0]  # Get the malignancy value
                    
                    
                    self.multi_class_labels.append(benign_value)

                    # print(f'The benign file name is: {patient_id_benign}')
                    # print(f'The benign_value file name is: {benign_value}')
                    self.file_paths.append(os.path.join(benign_folder, file_name))
                    self.binary_labels.append(0)  # Benign label

        if os.path.exists(malignant_folder):
            for file_name in os.listdir(malignant_folder):
                if file_name.endswith('.npy'):
                    patient_id_malign = file_name.split('_')[0]  # Extract case name
                    nodule_number_malign = file_name.split('_')[1]  # Extract nodule number

                    malign_value = df[
                        (df['patient_id'] == patient_id_malign) &
                        (df['nodule_number'] == nodule_number_malign)
                    ].malignancy.values[0]  # Get the malignancy value
                    self.multi_class_labels.append(malign_value)

                    # print(f'The malignant file name is: {patient_id_malign}')
                    # print(f'The malign_value file name is: {malign_value}')
                    self.file_paths.append(os.path.join(malignant_folder, file_name))
                    self.binary_labels.append(1)  # Malignant label

        logging.info(f"Loaded {len(self.file_paths)} samples from {self.dataset_type} folder.")

    def _print_nodule_counts(self) -> None:
        """Prints the count of benign and malignant nodules."""
        benign_count = self.binary_labels.count(0)
        malignant_count = self.binary_labels.count(1)
        logging.info(f"{self.dataset_type.capitalize()} Dataset: {benign_count} benign, {malignant_count} malignant nodules.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        binarys_label = self.binary_labels[idx]
        multi_class_label = self.multi_class_labels[idx]
        tensor = prepare_data(file_path)
        if tensor is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        return tensor, torch.tensor(multi_class_label).float() #torch.tensor(binarys_label).float(), file_path,


# -----------------------
# 8. PathName_MC_DatasetV4 Class
# -----------------------

class PathName_MC_DatasetV4(Dataset):
    """
    Combined Dataset class for managing train and test datasets but according to the path and the actual multi-class.

    Args:
        base_folder (str): Path to the base directory containing 'train' and 'test' folders.
    """
    def __init__(self, base_folder: str) -> None:
        self.datasets = {
            'train': CustomPathName_MC_DatasetV4(base_folder, dataset_type='train'),
            'test': CustomPathName_MC_DatasetV4(base_folder, dataset_type='test')
        }

    def __getitem__(self, dataset_type: str) -> Dataset:
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return self.datasets[dataset_type]
    

# -----------------------
# 9. CustomNLSTDatasetBenignV1 Class
# -----------------------

class CustomNLSTDatasetBenignV1(Dataset):
    """
    Dataset class for the NLST dataset version 1.

    Args:
        data_path (str): Path to the NLST dataset folder.
        fold (int): Fold number for cross-validation.
    """
    def __init__(self, folder_path: str, dataset_type: str = 'train') -> None:
        self.folder_path = os.path.join(folder_path, dataset_type)
        self.dataset_type = dataset_type
        self.binary_labels = []
        self.file_paths = []

        # Load the dataframe from folders
        self.df, self.malign = self._load_dataframe_from_folders()
        self._load_data_from_folders()
        self._print_nodule_counts()

    def _load_dataframe_from_folders(self) -> pd.DataFrame:
        benign_nodules = '/data/Datasets/Lungs/LUNA25-NLST/2D_patches_benign/'
        benign_folders = np.array([os.path.join(benign_nodules, folder) for folder in np.sort(os.listdir(benign_nodules))])[:91]
        benign_label = np.array([0 for _ in range(len(benign_folders))])

        benign_df = pd.DataFrame(benign_folders, columns=['folder'])
        benign_df['patient_id'] = np.sort(os.listdir(benign_nodules))[:91]
        benign_df['patient_id'] = benign_df['patient_id'].astype(int)
        benign_df['malignancy'] = benign_label

        chile_dataset = '/data/arumota/Nodules_Ohif/patch_chile/'
        path_folders = [os.path.join(chile_dataset, patient) for patient in sorted(os.listdir(chile_dataset))]

        chile_df = pd.read_csv('/data/arumota/PhD/Automation_bias/chile_malignancy.csv', sep=';')
        chile_df.rename(columns={'class': 'malignancy', 'patient': 'patient_id'}, inplace=True)
        chile_df['patient_id'] = chile_df['patient_id'].astype(int)

        #Add the paths on the dataframe
        chile_df['folder'] = path_folders

        # Keep only benign nodules from Chile Datasett
        self.malign = chile_df[chile_df['malignancy'] == 0]

        df_nlst_chile_merge = benign_df.merge(chile_df_benign, on=['patient_id'], how='outer')
        df_nlst_chile_merge.rename(columns={'folder_y': 'folder', 'malignancy_y': 'malignancy'}, inplace=True)

        # Merge folder_x and folder if folder_x is NaN
        df_nlst_chile_merge['folder'] = df_nlst_chile_merge['folder'].combine_first(df_nlst_chile_merge['folder_x'])
        df_nlst_chile_merge['malignancy'] = df_nlst_chile_merge['malignancy'].combine_first(df_nlst_chile_merge['malignancy_x'])
        del df_nlst_chile_merge['folder_x']; del df_nlst_chile_merge['malignancy_x']

        df_benign = pd.concat([df_nlst_chile_merge, benign_df], ignore_index=True)
        df_benign = df_benign.drop_duplicates(subset=['patient_id'], keep='first')

        # Obtain the malign nodules.
        chile_malign = chile_df[chile_df['malignancy'] == 1]

        malignant_nodules = '/data/Datasets/Lungs/LUNA25-NLST/2D_patches_malign/'
        malignant_folders = np.array([os.path.join(malignant_nodules, folder) for folder in np.sort(os.listdir(malignant_nodules))])[:91]
        malignant_label = np.array([1 for _ in range(len(malignant_folders))])

        #Dataframe with patient_id, malignant/benign label and path
        malignant_df = pd.DataFrame(malignant_folders, columns=['folder'])
        malignant_df['patient_id'] = np.sort(os.listdir(malignant_nodules))[:91]
        malignant_df['patient_id'] = malignant_df['patient_id'].astype(int)
        malignant_df['malignancy'] = malignant_label

        df_nlst_chile_merge_malign = malignant_df.merge(chile_malign, on=['patient_id'], how='outer')
        df_nlst_chile_merge_malign.rename(columns={'folder_y': 'folder', 'malignancy_y': 'malignancy'}, inplace=True)

        # Merge folder_x and folder if folder_x is NaN
        df_nlst_chile_merge_malign['folder'] = df_nlst_chile_merge_malign['folder'].combine_first(df_nlst_chile_merge_malign['folder_x'])
        df_nlst_chile_merge_malign['malignancy'] = df_nlst_chile_merge_malign['malignancy'].combine_first(df_nlst_chile_merge_malign['malignancy_x'])
        del df_nlst_chile_merge_malign['folder_x']; del df_nlst_chile_merge_malign['malignancy_x']

        df_malign = pd.concat([df_nlst_chile_merge_malign, malignant_df], ignore_index=True)
        df_malign = df_malign.drop_duplicates(subset=['patient_id'], keep='first')

        return df_benign, df_malign

    def _load_data_from_folders(self) -> None:
        """Loads file paths and binary_labels according to the dataframe."""
        # Make a training and testing sets with NLST and Chile dataset

        X_train, X_test, _, _ = train_test_split(self.df['patient_id'], self.df['malignancy'], test_size=0.2, random_state=42)
        # Concatenate to X_test the patient_id from the malignancy dataset
        X_test = np.concatenate((X_test, self.malign['patient_id'].values), axis=0)

        if self.dataset_type == 'train':
            self.df = self.df[self.df['patient_id'].isin(X_train)]
        elif self.dataset_type == 'test':
            self.df = self.df[self.df['patient_id'].isin(X_test)]
            # add the malignancy dataset to the test set
            self.df = pd.concat([self.df, self.malign[self.malign['patient_id'].isin(X_test)]], ignore_index=True)

        for index, patient in enumerate(self.df['patient_id']):
            patient_folder = self.df.loc[self.df['patient_id'] == patient, 'folder'].values[0]
            patient_malignancy = self.df.loc[self.df['patient_id'] == patient, 'malignancy'].values[0]

            self.binary_labels.append(patient_malignancy)
            list_folders = os.listdir(patient_folder)
            patient_folder = os.path.join(patient_folder, list_folders[0])

            image_folder = sorted(os.listdir(patient_folder))
            middle_value    = len(image_folder)//2

            tensor_image = image_folder[middle_value]
            self.file_paths.append(os.path.join(patient_folder, tensor_image))
            
        logging.info(f"Loaded NLST samples from {self.dataset_type} folder.")

    def _print_nodule_counts(self) -> None:
        """Prints the count of benign and malignant nodules."""
        benign_count = self.binary_labels.count(0)
        malignant_count = self.binary_labels.count(1)
        logging.info(f"{self.dataset_type.capitalize()} Dataset: {benign_count} benign, {malignant_count} malignant nodules.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        binarys_label = self.binary_labels[idx]
        tensor = prepare_data(file_path)
        if tensor is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        return tensor, torch.tensor(binarys_label).float() # torch.tensor(multi_class_label).float() #file_path,



# -----------------------
# 9. NLSTDatasetV1 Class
# -----------------------

class NLSTDatasetBenignV1(Dataset):
    """
    Combined Dataset class for managing train and test datasets.

    Args:
        base_folder (str): Path to the base directory containing 'train' and 'test' folders.
    """
    def __init__(self, base_folder: str) -> None:
        self.datasets = {
            'train': CustomNLSTDatasetBenignV1(base_folder, dataset_type='train'),
            'test': CustomNLSTDatasetBenignV1(base_folder, dataset_type='test')
        }

    def __getitem__(self, dataset_type: str) -> Dataset:
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return self.datasets[dataset_type]
    
# -----------------------
# 10. CustomLNDbDatasetBenignV1 Class
# -----------------------


class CustomLNDbDatasetBenignV1(Dataset):
    """
    Dataset class for the NLST dataset version 1.

    Args:
        data_path (str): Path to the NLST dataset folder.
        fold (int): Fold number for cross-validation.
    """
    def __init__(self, folder_path: str, dataset_type: str = 'train') -> None:
        self.folder_path = os.path.join(folder_path, dataset_type)
        self.dataset_type = dataset_type
        self.binary_labels = []
        self.file_paths = []

        # Load the dataframe from folders
        self.benign, self.malign = self._load_dataframe_from_folders()
        self._load_data_from_folders()
        self._print_nodule_counts()

    def _load_dataframe_from_folders(self) -> pd.DataFrame:
        LDNb_dataset = pd.read_csv("nodules_consensous_filtered3R.csv")
        # Change nodule column name to 'patient_id' for consistency
        LDNb_dataset.rename(columns={'nodule': 'patient_id'}, inplace=True)

        # Dataframe splitting
        df_malign = LDNb_dataset[LDNb_dataset['malignancy'] > 2.0].copy().reset_index(drop=True)
        df_benign = LDNb_dataset[LDNb_dataset['malignancy'] < 3.0].copy().reset_index(drop=True)

        # Change malignancy values to 1 for malignant and 0 for benign
        df_malign['malignancy'] = 1
        df_benign['malignancy'] = 0
        return df_benign, df_malign

    def _load_data_from_folders(self) -> None:
        """Loads file paths and binary_labels according to the dataframe."""
        LNDb_path = '/data/Datasets/Lungs/LNDb/LNDb_cropped_2D/'
        X_train, X_test, _, _ = train_test_split(self.benign['patient_id'], self.benign['malignancy'], 
                                                 test_size=0.8, random_state=42)
        print("Los datos benignos son", self.benign['patient_id'].groupby(self.benign['malignancy']).count())
        # Concatenate to X_test the patient_id from the malignancy dataset
        X_test = np.concatenate((X_test, self.malign['patient_id'].values), axis=0)


        if self.dataset_type == 'train':
            self.df = self.benign[self.benign['patient_id'].isin(X_train)]
        elif self.dataset_type == 'test':
            self.df = self.benign[self.benign['patient_id'].isin(X_test)]
            # add the malignancy dataset to the test set
            self.df = pd.concat([self.df, self.malign[self.malign['patient_id'].isin(X_test)]], ignore_index=True)

        # Load the nodule data
        self.file_paths = [os.path.join(LNDb_path, npy + ".npy") for npy in self.df['patient_id']]
        self.binary_labels = [self.df.loc[self.df['patient_id'] == patient, 'malignancy'].values[0] for patient in self.df['patient_id']]
        
            
        logging.info(f"Loaded LNDb samples from {self.dataset_type} folder.")

    def _print_nodule_counts(self) -> None:
        """Prints the count of benign and malignant nodules."""
        benign_count = self.binary_labels.count(0)
        malignant_count = self.binary_labels.count(1)
        logging.info(f"{self.dataset_type.capitalize()} Dataset: {benign_count} benign, {malignant_count} malignant nodules.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        binarys_label = self.binary_labels[idx]
        tensor = prepare_data(file_path)
        if tensor is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        return tensor, torch.tensor(binarys_label).float() # torch.tensor(multi_class_label).float() #file_path,



# -----------------------
# 11. LNDbDatasetBenignV1 Class
# -----------------------

class LNDbDatasetBenignV1(Dataset):
    """
    Combined Dataset class for managing train and test datasets.

    Args:
        base_folder (str): Path to the base directory containing 'train' and 'test' folders.
    """
    def __init__(self, base_folder: str) -> None:
        self.datasets = {
            'train': CustomLNDbDatasetBenignV1(base_folder, dataset_type='train'),
            'test': CustomLNDbDatasetBenignV1(base_folder, dataset_type='test')
        }

    def __getitem__(self, dataset_type: str) -> Dataset:
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return self.datasets[dataset_type]



# -----------------------
# 12. CustomNLSTDatasetBenign_ZeroFull Class
# -----------------------

class CustomNLSTDatasetBenign_ZeroFull(Dataset):
    """
    Dataset class for the NLST dataset version 1.

    Args:
        data_path (str): Path to the NLST dataset folder.
        fold (int): Fold number for cross-validation.
    """
    def __init__(self, folder_path: str, dataset_type: str = 'train') -> None:
        self.folder_path = os.path.join(folder_path, dataset_type)
        self.dataset_type = dataset_type
        self.binary_labels = []
        self.file_paths = []

        # Load the dataframe from folders
        self.df = self._load_dataframe_from_folders()
        self._load_data_from_folders()
        self._print_nodule_counts()

    def _load_dataframe_from_folders(self) -> pd.DataFrame:
        malignant_nodules = '/data/Datasets/Lungs/LUNA25-NLST/2D_patches_malign/'
        benign_nodules = '/data/Datasets/Lungs/LUNA25-NLST/2D_patches_benign/'
        malignant_folders = np.array([os.path.join(malignant_nodules, folder) for folder in np.sort(os.listdir(malignant_nodules))])[:91]
        benign_folders = np.array([os.path.join(benign_nodules, folder) for folder in np.sort(os.listdir(benign_nodules))])[:91]
        malignant_label = np.array([1 for _ in range(len(malignant_folders))])
        benign_label = np.array([0 for _ in range(len(benign_folders))])

        #Dataframe with patient_id, malignant/benign label and path
        malignant_df = pd.DataFrame(malignant_folders, columns=['folder'])
        malignant_df['patient_id'] = np.sort(os.listdir(malignant_nodules))[:91]
        malignant_df['patient_id'] = malignant_df['patient_id'].astype(int)
        malignant_df['malignancy'] = malignant_label


        benign_df = pd.DataFrame(benign_folders, columns=['folder'])
        benign_df['patient_id'] = np.sort(os.listdir(benign_nodules))[:91]
        benign_df['patient_id'] = benign_df['patient_id'].astype(int)
        benign_df['malignancy'] = benign_label

        nlst_df = pd.concat([malignant_df, benign_df], ignore_index=True)

        chile_dataset = '/data/arumota/Nodules_Ohif/patch_chile/'
        path_folders = [os.path.join(chile_dataset, patient) for patient in sorted(os.listdir(chile_dataset))]

        chile_df = pd.read_csv('/data/arumota/PhD/Automation_bias/chile_malignancy.csv', sep=';')
        chile_df.rename(columns={'class': 'malignancy', 'patient': 'patient_id'}, inplace=True)
        chile_df['patient_id'] = chile_df['patient_id'].astype(int)

        #Add the paths on the dataframe
        chile_df['folder'] = path_folders

        df_nlst_chile_merge = nlst_df.merge(chile_df, on=['patient_id'], how='outer')
        df_nlst_chile_merge.rename(columns={'folder_y': 'folder', 'malignancy_y': 'malignancy'}, inplace=True)

        # Merge folder_x and folder if folder_x is NaN
        df_nlst_chile_merge['folder'] = df_nlst_chile_merge['folder'].combine_first(df_nlst_chile_merge['folder_x'])
        df_nlst_chile_merge['malignancy'] = df_nlst_chile_merge['malignancy'].combine_first(df_nlst_chile_merge['malignancy_x'])
        del df_nlst_chile_merge['folder_x']; del df_nlst_chile_merge['malignancy_x']

        df = pd.concat([df_nlst_chile_merge, nlst_df], ignore_index=True)
        df = df.drop_duplicates(subset=['patient_id'], keep='first')
        return df

    def _load_data_from_folders(self) -> None:
        """Loads file paths and binary_labels according to the dataframe."""
        # Make a training and testing sets with NLST and Chile dataset

        X_train, X_test, y_train, _ = train_test_split(self.df['patient_id'], self.df['malignancy'], 
                                                 test_size=0.10, random_state=42)
        X_train, X_train_test, _, _ = train_test_split(X_train, y_train, 
                                                       train_size=0.8, random_state=42)

        if self.dataset_type == 'train':
            self.df = self.df[self.df['patient_id'].isin(X_train)]
        elif self.dataset_type == 'test':
            self.df = self.df[self.df['patient_id'].isin(X_test)]

        chile_dataset = "/data/arumota/Nodules_Ohif/patch_chile"
        for index, patient in enumerate(self.df['patient_id']):
            patient_folder = self.df.loc[self.df['patient_id'] == patient, 'folder'].values[0]
            patient_malignancy = self.df.loc[self.df['patient_id'] == patient, 'malignancy'].values[0]

            self.binary_labels.append(patient_malignancy)
            if patient_folder.startswith(chile_dataset):
                patient_folder = os.path.join(patient_folder, 'N_0')
            else:
                list_folders = os.listdir(patient_folder)
                patient_folder = os.path.join(patient_folder, list_folders[0])

            image_folder = sorted(os.listdir(patient_folder))
            middle_value    = len(image_folder)//2

            tensor_image = image_folder[middle_value]
            self.file_paths.append(os.path.join(patient_folder, tensor_image))
            

        logging.info(f"Loaded NLST samples from {self.dataset_type} folder.")

    def _print_nodule_counts(self) -> None:
        """Prints the count of benign and malignant nodules."""
        benign_count = self.binary_labels.count(0)
        malignant_count = self.binary_labels.count(1)
        logging.info(f"{self.dataset_type.capitalize()} Dataset: {benign_count} benign, {malignant_count} malignant nodules.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        binarys_label = self.binary_labels[idx]
        tensor = prepare_data(file_path)
        if tensor is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        return tensor, torch.tensor(binarys_label).float() # torch.tensor(multi_class_label).float() #file_path,



# -----------------------
# 13. NLSTDatasetBenign_ZeroFull Class
# -----------------------

class NLSTDatasetBenign_ZeroFull(Dataset):
    """
    Combined Dataset class for managing train and test datasets.

    Args:
        base_folder (str): Path to the base directory containing 'train' and 'test' folders.
    """
    def __init__(self, base_folder: str) -> None:
        self.datasets = {
            'train': CustomNLSTDatasetBenign_ZeroFull(base_folder, dataset_type='train'),
            'test': CustomNLSTDatasetBenign_ZeroFull(base_folder, dataset_type='test')
        }

    def __getitem__(self, dataset_type: str) -> Dataset:
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return self.datasets[dataset_type]

# -----------------------
# 14. Custom_LIDC_malignant_dataset Class
# -----------------------
class CustomLIDC_malignant_dataset(Dataset):
    """
    Dataset class for the LIDC in terms of a malignant reconstruction.

    Args:
        data_path (str): Path to the LIDC dataset folder.
        fold (int): Fold number for cross-validation.
    """
    def __init__(self, folder_path: str, dataset_type: str = 'train') -> None:
        self.folder_path = os.path.join(folder_path, dataset_type)
        self.dataset_type = dataset_type
        self.binary_labels = []
        self.file_paths = []

        # Load the dataframe from folders
        self.benign, self.malign = self._load_dataframe_from_folders()
        self._load_data_from_folders()
        self._print_nodule_counts()

    def _load_dataframe_from_folders(self) -> pd.DataFrame:
        csv_path = "/data/arumota/PhD/Pasantia/project2.0_mmd/dataset_LIDC_all_best_matches_2612.csv"
        data = pd.read_csv(csv_path)
    
        # Filter the dataframe to keep only nodules with more than 3.0 radiologist
        data = data[
            ~(
                (data['radiologist'] < 3.0) 
            )
        ].reset_index(drop=True)
        # Filter the dataframe to keep nodules that are not indeterminated (malignancy_agg = 3.0)
        data = data[
            ~(
                (data['malignancy_agg'] == 3.0) &
                (data['malignancy_agg'] == 3)
            )
        ].reset_index(drop=True)

        # Reemplazar los valores del dataframe para que 1.0 y 2.0 sean 0 (benignos), 3.0 sea -1 (indeterminado) y 4.0 y 5.0 sean 1 (malignos)
        for r in ["R1", "R2", "R3", "R4", "MMV", "malignancy_agg"]:
            if r in data.columns:
                data[r] = data[r].replace({1.0: 0, 2.0: 0, 3.0: -1, 4.0: 1, 5.0: 1})

        # self.data['split'] = self.data['label_type'].apply(lambda x: 'train' if x in ['Clean agreement'] else 'test')
        #Dependiendo del label_type devolver los conteos en el split
        # print(f"Dataset label_type counts by split:\n{self.data.groupby('split')['label_type'].value_counts()}")
        malign_data = data[data['malignancy_agg'] == 1].copy().reset_index(drop=True)
        benign_data = data[data['malignancy_agg'] == 0].copy().reset_index(drop=True)
        return benign_data, malign_data

    def _load_data_from_folders(self) -> None:
        """Loads file paths and binary_labels according to the dataframe."""
        # Do a train-test split of the dataframe based on best_match and malignancy_agg, keeping the same proportion of benign and malignant nodules in both sets.
        X_train, X_test, y_train, _ = train_test_split(self.malign['best_match'], self.malign['malignancy_agg'], 
                                                 test_size=0.2, random_state=42)

        X_train_benign, X_test_benign, _, _ = train_test_split(self.benign['best_match'], self.benign['malignancy_agg'],
                                                    test_size=0.5, random_state=42)
        # print("\nLos datos benignos son", self.benign['best_match'].groupby(self.benign['malignancy_agg']).count())
        # print("Los datos malignos son", self.malign['best_match'].groupby(self.malign['malignancy_agg']).count())

        # Concatenate to X_test the best_match from the benign dataset
        X_train = np.concatenate((X_train, X_train_benign), axis=0)
        X_test = np.concatenate((X_test, X_test_benign), axis=0)

        # Depending on the dataset type, filter the dataframe to keep only the samples in the train or test set.
        if self.dataset_type == 'train':
            self.df = self.malign[self.malign['best_match'].isin(X_train)].reset_index(drop=True)
            self.df = pd.concat([self.df, self.benign[self.benign['best_match'].isin(X_train)]], ignore_index=True).reset_index(drop=True)

        elif self.dataset_type == 'test':
            self.df = self.malign[self.malign['best_match'].isin(X_test)].reset_index(drop=True)
            # Add the malignancy dataset to the test set
            self.df = pd.concat([self.df, self.benign[self.benign['best_match'].isin(X_test)]], ignore_index=True).reset_index(drop=True)

        # Time to load the nodule data according to the dataframe, using the best_match column to find the corresponding .npy file in the LIDC dataset. The best_match column contains the name of the .npy file with the extension
        self.file_paths = self.df['best_match']
        self.binary_labels = self.df['malignancy_agg'].values

        print(f"We have {len(self.file_paths)} samples in the {self.dataset_type} set, with {np.sum(self.binary_labels == 0)} benign and {np.sum(self.binary_labels == 1)} malignant nodules.")
        logging.info(f"Loaded LIDC malignant samples from {self.dataset_type} folder.")


    def _print_nodule_counts(self) -> None:
        """Prints the count of benign and malignant nodules."""
        benign_count = np.sum(self.binary_labels == 0)
        malignant_count = np.sum(self.binary_labels == 1)
        logging.info(f"{self.dataset_type.capitalize()} Dataset: {benign_count} benign, {malignant_count} malignant nodules.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths.iloc[idx]
        binarys_label = self.binary_labels[idx]
        tensor = prepare_data(file_path)
        if tensor is None:
            raise FileNotFoundError(f"Failed to load {file_path}")
        return tensor, torch.tensor(binarys_label).float() # torch.tensor(multi_class_label).float() #file_path,

# -----------------------
# 15. LIDC_malignant_dataset Class
# -----------------------

class LIDC_malignant_dataset(Dataset):
    """
    Combined Dataset class for managing train and test datasets.

    Args:
        base_folder (str): Path to the base directory containing 'train' and 'test' folders.
    """
    def __init__(self, base_folder: str) -> None:
        self.datasets = {
            'train': CustomLIDC_malignant_dataset(base_folder, dataset_type='train'),
            'test': CustomLIDC_malignant_dataset(base_folder, dataset_type='test')
        }

    def __getitem__(self, dataset_type: str) -> Dataset:
        if dataset_type not in self.datasets:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
        return self.datasets[dataset_type]


# -----------------------
# 12. Example Usage
# -----------------------

# Example usage
# dataset = create_dataset('v3')
# print(f"Loaded {len(dataset['train'])} train samples and {len(dataset['test'])} test samples.")