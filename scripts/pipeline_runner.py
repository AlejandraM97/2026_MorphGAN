import os
import sys
import logging
import traceback
import hydra
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Local imports (use PYTHONPATH or relative structure)
from config import registry
from models.base.train import train_ganlung
from train.test import GANTester
from models.base.network import GANLung
from utils.metrics import format_metrics
from data.data import create_dataset  # for LIDC
from data.hist_loader import create_hist_loader  # for Hist

# --------------------
# Logging setup
# --------------------
log_path = '../results/logs'
os.makedirs(log_path, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_path, 'pipeline.log'))
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

# --------------------
# Hydra Opt Wrapper
# --------------------
class Opt:
    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Opt(value))
            else:
                setattr(self, key, value)

# --------------------
# Dataset Loader
# --------------------
def create_lidc_loaders(folder_path, batchsize, version):
    logger.info(f"Creating LIDC dataset for version: {version}")
    dataset = create_dataset(version)
    train_loader = DataLoader(dataset['train'], batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset['test'], batch_size=batchsize, shuffle=False)
    return train_loader, test_loader

@hydra.main(config_path="../config", config_name="pipeline_config", version_base=None)
def run(cfg: DictConfig):
    logger.info("Starting training + evaluation pipeline...")

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    opt = Opt(flat_cfg)
    registry.opt = opt

    wandb.init(
        project="ganlung-pipeline",
        name=f"{opt.model_name}_w{opt.w_mmd}_{opt.dataset_version}",
        config=flat_cfg
    )

    try:
        # --- Primary Dataset: LIDC ---
        train_loader, test_loader = create_lidc_loaders(opt.folder_path, opt.batchsize, opt.dataset_version)

        # --- Secondary Dataset: Hist ---
        hist_loader = create_hist_loader(
            opt.secondary_dataset.hist_csv,
            opt.secondary_dataset.npy_dir,
            batch_size=opt.secondary_dataset.batch_size
        )

        # --- Training ---
        logger.info(f"Training model: {opt.model_name}")
        train_ganlung(opt, train_loader, test_loader)

        # --- Evaluation on LIDC ---
        model = GANLung(opt)
        model.seed(opt.seed)
        weight_path = os.path.join(opt.save_path, f"{opt.model_name}.pth")
        model.load_model(weight_path)

        tester = GANTester(model, train_loader, test_loader, model.device)
        tester.alpha = getattr(opt, "alpha", 0.8)

        thresholds = tester.get_threshold_precision_recall(train_loader)
        metrics, y_true, scores = tester.test()
        classification_results, _ = tester.calculate_classification_metrics(thresholds, metrics, y_true, scores)

        for method_name, m in classification_results.items():
            wandb.log({
                f"lidc/{method_name}/specificity": m["specificity"],
                f"lidc/{method_name}/auc": m["auc"],
                f"lidc/{method_name}/f1": m["f1"],
                f"lidc/{method_name}/precision": m["precision"],
                f"lidc/{method_name}/recall": m["recall"]
            })

        logger.info("Metrics for LIDC dataset:")
        format_metrics(classification_results)

        # --- Evaluation on HIST ---
        hist_tester = GANTester(model, train_loader, hist_loader, model.device)
        hist_tester.alpha = tester.alpha  # reuse same alpha

        thresholds = hist_tester.get_threshold_precision_recall(train_loader)
        metrics, y_true, scores = hist_tester.test()
        hist_results, _ = hist_tester.calculate_classification_metrics(thresholds, metrics, y_true, scores)

        for method_name, m in hist_results.items():
            wandb.log({
                f"hist/{method_name}/specificity": m["specificity"],
                f"hist/{method_name}/auc": m["auc"],
                f"hist/{method_name}/f1": m["f1"],
                f"hist/{method_name}/precision": m["precision"],
                f"hist/{method_name}/recall": m["recall"]
            })

        logger.info("Metrics for HIST dataset:")
        format_metrics(hist_results)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Pipeline completed.")

if __name__ == "__main__":
    run()