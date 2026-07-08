import os
import torch
import logging
import pandas as pd
from torch.utils.data import DataLoader
from models.morph.network import GANLung
from utils.losses import plot_losses, generate_random_number, store_results, plot_histograms
from train.test import GANTester
from config.registry import opt

# Configure logging to save logs in the results/logs directory
log_path = './results/logs'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'training.log')),
        logging.StreamHandler()
    ]
)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------
# 1. Helper Functions
# -----------------------

def save_model(model: GANLung, path: str) -> None:
    """Saves the GAN model."""
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        model.save_model(path)
        logging.info(f"Model saved at {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

def track_loss(all_losses: dict, loss_values: dict) -> None:
    """Tracks and stores loss values."""
    for key in all_losses.keys():
        all_losses[key].append(loss_values.get(key, 0))

def log_epoch_status(epoch: int, step: int, opt, loss_values: dict) -> None:
    """Logs the status of each epoch."""
    logging.info(
        f"Epoch [{epoch}/{opt.epochs}], Step [{step}], "
        f"G Loss: {loss_values['err_g']:.5f}, D Loss: {loss_values['err_d']:.5f}, "
        f"G Con Loss: {loss_values['err_g_con']:.5f}, G Adv Loss: {loss_values['err_g_adv']:.5f}, "
        f"G Enc Loss: {loss_values['err_g_enc']:.5f}"
    )

# -----------------------
# 2. Main Training Function
# -----------------------

def train(opt, train_loader: DataLoader, test_loader: DataLoader) -> None:
    lungan = GANLung(opt)
    # Load the model if a path is provided
    # lungan.load_model(opt.load_path)
    all_losses = {k: [] for k in ['err_g', 'err_d', 'err_g_con', 'err_g_adv', 'err_g_enc']}
    
    # Get the directory of the current file (train.py)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # dos niveles arriba
    model_save_path = os.path.join(project_root, "results", "weights")

    

    os.makedirs(model_save_path, exist_ok=True)  # Ensure the directory exists

    # Training loop
    for epoch in range(opt.epochs):
        epoch_g_loss, epoch_d_loss, epoch_g_con_loss, epoch_g_adv_loss, epoch_g_enc_loss, epoch_mmd_loss = 0, 0, 0, 0, 0, 0  # Track total loss for the epoch
        steps = 0  # Track number of steps

        for i, data in enumerate(train_loader):
            data, labels = data

            # Filter out malignant nodules (train only on benign)
            benign_mask = labels == 0
            data = data[benign_mask]
            labels = labels[benign_mask]

            if len(data) > 0:
                input = data.to(lungan.device)
            else:
                continue  # Skip if no benign samples

            # Optimize parameters
            err_g, err_d, err_g_con, err_g_adv, err_g_enc, mmd = lungan.optimize_params(input)

            # Accumulate losses for the epoch
            epoch_g_loss += err_g.item()
            epoch_d_loss += err_d.item()
            epoch_g_con_loss += err_g_con.item()
            epoch_g_adv_loss += err_g_adv.item()
            epoch_g_enc_loss += err_g_enc.item()
            epoch_mmd_loss += mmd.item()
            steps += 1

            # Track losses 
            loss_values = {
                'err_g': err_g.item(),
                'err_d': err_d.item(),
                'err_g_con': err_g_con.item(),
                'err_g_adv': err_g_adv.item(),
                'err_g_enc': err_g_enc.item(),
                'mmd': mmd.item()
            }
            track_loss(all_losses, loss_values)

        # Log only once per epoch
        logging.info(
            f"Epoch [{epoch + 1}/{opt.epochs}] - "
            f"G Loss: {epoch_g_loss / steps:.5f}, "
            f"D Loss: {epoch_d_loss / steps:.5f}, "
            f"G Con Loss: {epoch_g_con_loss / steps:.5f}, "
            f"G Adv Loss: {epoch_g_adv_loss / steps:.5f}, "
            f"G Enc Loss: {epoch_g_enc_loss / steps:.5f}, "
            f"MMD Loss: {epoch_mmd_loss / steps:.5f}"
        )

    # Save model only after full training
    lungan.save_model(model_save_path, model_name=opt.model_name)
    logging.info(f"Model weights saved at {model_save_path}/{opt.model_name}.pth")

    # Plot and save losses
    plot_losses(all_losses, opt.save_path)
    logging.info("Training completed. Generating plots and saving results.")
