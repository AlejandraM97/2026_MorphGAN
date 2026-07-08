import sys
import os

# Add the project root (one level up from scripts/) to sys.path
sys.path.append(os.path.abspath('/data/arumota/PhD/Pasantia/project2.0_mmd'))

import logging
import importlib
import traceback
from torch.utils.data import DataLoader

# Config and setup
from config.options import Options
from config import registry
from data.data import create_dataset

# Load options
opt = Options().parse()
registry.opt = opt

# -----------------------
# Logging Configuration
# -----------------------

log_path = '../../results/logs'
os.makedirs(log_path, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(log_path, 'run.log'))
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - Line %(lineno)d - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -----------------------
# 1. Helper Functions
# -----------------------

def dynamic_import(module_path: str, obj_name: str):
    """
    Imports an object dynamically from a module path.
    """
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def handle_training(opt, train_loader: DataLoader, test_loader: DataLoader) -> None:
    """
    Dynamically imports and runs the appropriate training function.
    """
    model_type = opt.model_type  # e.g., 'ganlung', 'classifier', etc.
    train_module = f"models.{model_type}.train"

    try:
        train_func = dynamic_import(train_module, "train")
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load train function from {train_module}: {e}")

    logging.info(f"Using training module: {train_module}")
    train_func(opt, train_loader, test_loader)

def create_data_loaders(dataset_version: str) -> tuple[DataLoader, DataLoader]:
    dataset = create_dataset(dataset_version)
    train_loader = DataLoader(dataset['train'], batch_size=opt.batchsize, shuffle=True)
    test_loader = DataLoader(dataset['test'], batch_size=opt.batchsize, shuffle=False)
    return train_loader, test_loader

# -----------------------
# 2. Main Function
# -----------------------

def run() -> None:
    logging.info("Starting the run function...")
    try:
        train_loader, test_loader = create_data_loaders(opt.dataset_version)
        logging.info("Data loaders created successfully.")
        handle_training(opt, train_loader, test_loader)
        logging.info("Training and evaluation completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.error(traceback.format_exc())
    except ValueError as e:
        logging.error(f"Value error: {e}")
        logging.error(traceback.format_exc())
    except KeyError as e:
        logging.error(f"Key error: {e}")
        logging.error(traceback.format_exc())
    except TypeError as e:
        logging.error(f"Type error: {e}")
        logging.error(traceback.format_exc())
    except AttributeError as e:
        logging.error(f"Attribute error: {e}")
        logging.error(traceback.format_exc())
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("Run function has completed.")

if __name__ == "__main__":
    run()
