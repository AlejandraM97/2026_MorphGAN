import os
import secrets
import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from math import exp
from config.registry import opt
import seaborn as sns

# Configure logging
log_path = './results/logs'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'utils.log')),
        logging.StreamHandler()
    ]
)


# -----------------------
# 1. Helper Functions
# -----------------------

def save_plot(fig, path: str) -> None:
    """Saves a plot to the specified path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        logging.info(f"Plot saved at {path}")
    except Exception as e:
        logging.error(f"Failed to save plot at {path}: {e}")

def generate_random_number() -> str:
    """Generates an 8-digit random number using the secrets module."""
    return ''.join([str(secrets.randbelow(10)) for _ in range(8)])

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """Saves a DataFrame to CSV."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"CSV saved at {path}")
    except Exception as e:
        logging.error(f"Failed to save CSV at {path}: {e}")

def calculate_metrics(malignant, benign, y_true, y_scores, threshold) -> dict:
    """Calculates performance metrics for a given threshold."""
    trueP = sum(idx > threshold for idx in malignant)
    falseN = len(malignant) - trueP
    trueN = sum(idx <= threshold for idx in benign)
    falseP = len(benign) - trueN
    precision = trueP / (trueP + falseP) if (trueP + falseP) > 0 else 0
    recall = trueP / (trueP + falseN) if (trueP + falseN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (trueP + trueN) / (trueN + falseP + trueP + falseN)
    specificity = trueN / (trueN + falseP) if (trueN + falseP) > 0 else 0
    auc_score = roc_auc_score(y_true, y_scores)

    return {
        'TP': trueP,
        'FP': falseP,
        'TN': trueN,
        'FN': falseN,
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'AUC': auc_score
    }

def store_results(results, custom, percentile):
    # Extract the desired values from the mse and sum_mse results dictionaries
    data = {
        "info": [custom],
        "thr": [percentile],
        
        # Metrics for mse_scores
        'AUC_mse': [results['mse_metrics']['AUC']],
        'TP_mse': [results['mse_metrics']['TP']],
        'FP_mse': [results['mse_metrics']['FP']],
        'TN_mse': [results['mse_metrics']['TN']],
        'FN_mse': [results['mse_metrics']['FN']],
        'Precision_mse': [results['mse_metrics']['Precision']],
        'Recall_mse': [results['mse_metrics']['Recall']],
        'F1_mse': [results['mse_metrics']['F1']],
        'Accuracy_mse': [results['mse_metrics']['Accuracy']],
        'Specificity_mse': [results['mse_metrics']['Specificity']],
        
        # Metrics for sum_mse_scores
        'AUC_sum': [results['sum_metrics']['AUC']],
        'TP_sum': [results['sum_metrics']['TP']],
        'FP_sum': [results['sum_metrics']['FP']],
        'TN_sum': [results['sum_metrics']['TN']],
        'FN_sum': [results['sum_metrics']['FN']],
        'Precision_sum': [results['sum_metrics']['Precision']],
        'Recall_sum': [results['sum_metrics']['Recall']],
        'F1_sum': [results['sum_metrics']['F1']],
        'Accuracy_sum': [results['sum_metrics']['Accuracy']],
        'Specificity_sum': [results['sum_metrics']['Specificity']],
    }

    df = pd.DataFrame(data)

    return df

# -----------------------
# 2. Plotting Functions
# -----------------------

def plot_losses(all_losses: dict, save_path: str) -> None:
    """Plots and saves training losses."""
    fig, axs = plt.subplots(len(all_losses), 1, figsize=(10, 15))
    for idx, (loss_name, loss_values) in enumerate(all_losses.items()):
        axs[idx].plot(loss_values, label=loss_name.replace('_', ' ').title())
        axs[idx].set_title(f'{loss_name.replace("_", " ").title()} over Iterations')
        axs[idx].legend()
        axs[idx].grid(True)
    plt.tight_layout()
    save_plot(fig, os.path.join(save_path, 'plots', 'losses.png'))

def plot_histograms(results: dict, save_path: str) -> None:
    """Plots histograms for MSE scores."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    sns.histplot(results['mse_0'], bins=30, kde=True, color='blue', ax=axes[0], label='Malignancy == 0')
    sns.histplot(results['mse_1'], bins=30, kde=True, color='red', ax=axes[0], label='Malignancy == 1')
    axes[0].set_title('MSE Histogram (Image Reconstructions)')
    axes[0].legend()
    plt.tight_layout()
    save_plot(fig, os.path.join(save_path, 'plots', 'histograms.png'))

# -----------------------
# 3. Loss Functions
# -----------------------

def l2_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes L2 loss."""
    return torch.mean((input - target) ** 2)

def l2_z_loss(input, target):
    """Computes L2 loss."""
    return np.linalg.norm(input.cpu().detach().numpy() - target.cpu().detach().numpy())

def ssim_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes SSIM-based loss."""
    return 1 - ssim(input, target)

def combined_loss(input: torch.Tensor, target: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Computes a combined SSIM and MSE loss."""
    ssim_val = ssim(input, target)
    mse_val = F.mse_loss(input, target)
    return alpha * (1 - ssim_val) + (1 - alpha) * mse_val

import torch
from skimage import morphology

class MorphologicalPixelLoss:
    def __init__(self, kernel_size=3):
        self.kernel = morphology.disk(kernel_size)
        # self.kernel = morphology.star(kernel_size)  

    def __call__(self, fake, real):
        device = real.device

        # Apertura morfológica
        real_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), self.kernel) for img in real]
        fake_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), self.kernel) for img in fake]

        # Volver a tensores
        real_openings = torch.stack([torch.tensor(img, device=device) for img in real_openings])
        fake_openings = torch.stack([torch.tensor(img, device=device) for img in fake_openings])

        # Top-hat transform
        tophat_real = (real - real_openings).detach()
        tophat_fake = (fake - fake_openings).detach()

        # Mapa espacial: suma de energía por pixel
        weight_map = (tophat_real ** 2 + tophat_fake ** 2)
        # weight_map = (tophat_real + tophat_fake)

        # weight_map = (real_openings - fake_openings).detach()**2
        # Weighted MSE pixel a pixel
        diff_squared = (real - fake) ** 2
        return torch.mean(weight_map * diff_squared)

class MorphologicalLoss:
    def __init__(self, kernel_size=3):
        self.kernel = morphology.disk(kernel_size)

    def __call__(self, fake, real):
        # Mover a CPU para usar skimage
        fake_np = [morphology.opening(img.squeeze().detach().cpu().numpy(), self.kernel) for img in fake]
        real_np = [morphology.opening(img.squeeze().detach().cpu().numpy(), self.kernel) for img in real]

        # Convertir a tensores en el mismo dispositivo
        device = real.device
        fake_opening = torch.stack([torch.tensor(img, device=device) for img in fake_np])
        real_opening = torch.stack([torch.tensor(img, device=device) for img in real_np])

        # Top-hat transform
        tophat_fake = fake - fake_opening
        tophat_real = real - real_opening

        # Weighted MSE
        diff_squared = ((real - fake) ** 2)
        weights = (tophat_fake.norm() + tophat_real.norm()).detach()

        return torch.mean(torch.sum(diff_squared * weights))
    

class MorphologicalLossTest:
    def __init__(self, kernel_size=3):
        self.kernel = morphology.disk(kernel_size)

    def __call__(self, fake, real):

        fake_np = [morphology.opening(img.squeeze(), self.kernel) for img in fake]
        real_np = [morphology.opening(img.squeeze(), self.kernel) for img in real]

        fake_opening = torch.stack([torch.tensor(img) for img in fake_np])
        real_opening = torch.stack([torch.tensor(img) for img in real_np])

        fake_img = torch.tensor(fake)
        real_img = torch.tensor(real)


        # Top-hat transform
        tophat_fake = fake_img - fake_opening
        tophat_real = real_img - real_opening

        # Weighted MSE
        diff_squared = ((real_img - fake_img) ** 2)
        weights = (tophat_fake.norm() + tophat_real.norm()).detach()

        return torch.mean(torch.sum(diff_squared * weights))

class LatentChannelStatsLoss:
    def __init__(self, reduction="none"):
        self.reduction = reduction

    def __call__(self, z, z_hat):
        """
        Compares per-channel mean and std between z and z_hat
        z, z_hat: tensors of shape [B, C, H, W]
        """
        z_mean = z.mean(dim=[2, 3])      # [B, C]
        z_std = z.std(dim=[2, 3])        # [B, C]
        z_hat_mean = z_hat.mean(dim=[2, 3])
        z_hat_std = z_hat.std(dim=[2, 3])

        loss_mean = F.mse_loss(z_mean, z_hat_mean, reduction=self.reduction)
        loss_std = F.mse_loss(z_std, z_hat_std, reduction=self.reduction)

        return loss_mean + loss_std
    
class SobelMap:
    def __init__(self):
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def __call__(self, fake, real):
        device = real.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # Gradientes de real y fake
        grad_real = torch.sqrt(F.conv2d(real, sobel_x, padding=1) ** 2 +
                               F.conv2d(real, sobel_y, padding=1) ** 2).detach()

        grad_fake = torch.sqrt(F.conv2d(fake, sobel_x, padding=1) ** 2 +
                               F.conv2d(fake, sobel_y, padding=1) ** 2).detach()

        weight_map = grad_real + grad_fake
        diff_squared = (real - fake) ** 2

        return grad_real, grad_fake, weight_map * diff_squared

class SobelPixelLoss:
    def __init__(self):
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def __call__(self, fake, real):
        device = real.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        # Gradientes de real y fake
        grad_real = torch.sqrt(F.conv2d(real, sobel_x, padding=1) ** 2 +
                               F.conv2d(real, sobel_y, padding=1) ** 2).detach()

        grad_fake = torch.sqrt(F.conv2d(fake, sobel_x, padding=1) ** 2 +
                               F.conv2d(fake, sobel_y, padding=1) ** 2).detach()

        weight_map = grad_real + grad_fake
        diff_squared = (real - fake) ** 2

        return torch.mean(weight_map * diff_squared)

class SobelLoss:
    def __init__(self):
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def __call__(self, fake, real):
        device = real.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        grad_x_fake = F.conv2d(fake, sobel_x, padding=1)
        grad_y_fake = F.conv2d(fake, sobel_y, padding=1)
        grad_fake = torch.sqrt(grad_x_fake ** 2 + grad_y_fake ** 2)

        grad_x_real = F.conv2d(real, sobel_x, padding=1)
        grad_y_real = F.conv2d(real, sobel_y, padding=1)
        grad_real = torch.sqrt(grad_x_real ** 2 + grad_y_real ** 2)

        weight = (grad_fake + grad_real).detach()
        diff = ((real - fake) ** 2)

        return torch.mean(torch.sum(diff * weight))
    
class SobelLossTest:
    def __init__(self):
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def __call__(self, fake, real):
        device = 'cuda'
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)
        # Gradientes de real y fake
        fake = torch.tensor(fake, device=device)
        real = torch.tensor(real, device=device)
        
        grad_x_fake = F.conv2d(fake, sobel_x, padding=1)
        grad_y_fake = F.conv2d(fake, sobel_y, padding=1)
        grad_fake = torch.sqrt(grad_x_fake ** 2 + grad_y_fake ** 2)

        grad_x_real = F.conv2d(real, sobel_x, padding=1)
        grad_y_real = F.conv2d(real, sobel_y, padding=1)
        grad_real = torch.sqrt(grad_x_real ** 2 + grad_y_real ** 2)

        weight = (grad_fake + grad_real).detach()
        diff = ((real - fake) ** 2)

        return torch.mean(torch.sum(diff * weight))
    
    
class CombinedSpatialLoss:
    def __init__(self, morph_weight=0.5, sobel_weight=0.5, kernel_size=3):
        self.morph_loss = MorphologicalPixelLoss(kernel_size)
        self.sobel_loss = SobelPixelLoss()
        self.w_morph = morph_weight
        self.w_sobel = sobel_weight

    def __call__(self, fake, real):
        device = real.device

        # === MORPHOLOGICAL MAP (copied from MorphologicalPixelLoss)
        kernel = morphology.disk(self.morph_loss.kernel.shape[0] // 2)
        real_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in real]
        fake_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in fake]
        real_openings = torch.stack([torch.tensor(img, device=device) for img in real_openings])
        fake_openings = torch.stack([torch.tensor(img, device=device) for img in fake_openings])
        tophat_real = (real - real_openings).detach()
        tophat_fake = (fake - fake_openings).detach()
        morph_map = tophat_real ** 2 + tophat_fake ** 2

        # === SOBEL MAP (copied from SobelPixelLoss)
        sobel_x = self.sobel_loss.sobel_x.to(device)
        sobel_y = self.sobel_loss.sobel_y.to(device)
        grad_real = torch.sqrt(F.conv2d(real, sobel_x, padding=1) ** 2 +
                               F.conv2d(real, sobel_y, padding=1) ** 2).detach()
        grad_fake = torch.sqrt(F.conv2d(fake, sobel_x, padding=1) ** 2 +
                               F.conv2d(fake, sobel_y, padding=1) ** 2).detach()
        sobel_map = grad_real + grad_fake

        # === Combined Weight Map and Loss
        combined_map = self.w_morph * morph_map + self.w_sobel * sobel_map
        diff_squared = (real - fake) ** 2
        return torch.mean(combined_map * diff_squared)
        

class CombinedSpatialLossTest:
    def __init__(self, morph_weight=0.5, sobel_weight=0.5, kernel_size=3):
        self.morph_loss = MorphologicalPixelLoss(kernel_size)
        self.sobel_loss = SobelPixelLoss()
        self.w_morph = morph_weight
        self.w_sobel = sobel_weight

    def __call__(self, fake, real):
        device = real.device

        # === MORPHOLOGICAL MAP (copied from MorphologicalPixelLoss)
        kernel = morphology.disk(self.morph_loss.kernel.shape[0] // 2)
        real_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in real]
        fake_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in fake]
        real_openings = torch.stack([torch.tensor(img, device=device) for img in real_openings])
        fake_openings = torch.stack([torch.tensor(img, device=device) for img in fake_openings])
        tophat_real = (real - real_openings).detach()
        tophat_fake = (fake - fake_openings).detach()
        morph_map = tophat_real ** 2 + tophat_fake ** 2

        # === SOBEL MAP (copied from SobelPixelLoss)
        sobel_x = self.sobel_loss.sobel_x.to(device)
        sobel_y = self.sobel_loss.sobel_y.to(device)
        grad_real = torch.sqrt(F.conv2d(real, sobel_x, padding=1) ** 2 +
                               F.conv2d(real, sobel_y, padding=1) ** 2).detach()
        grad_fake = torch.sqrt(F.conv2d(fake, sobel_x, padding=1) ** 2 +
                               F.conv2d(fake, sobel_y, padding=1) ** 2).detach()
        sobel_map = grad_real + grad_fake

        # === Combined Weight Map and Loss
        combined_map = self.w_morph * morph_map + self.w_sobel * sobel_map
        diff_squared = (real - fake) ** 2
        return torch.mean(torch.sum(combined_map * diff_squared))

class SummedSpatialLoss:
    def __init__(self, kernel_size=3):
        self.morph_loss = MorphologicalPixelLoss(kernel_size)
        self.sobel_loss = SobelPixelLoss()

    def __call__(self, fake, real):
        device = real.device

        # === MORPHOLOGICAL MAP (same as before)
        kernel = morphology.disk(self.morph_loss.kernel.shape[0] // 2)
        real_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in real]
        fake_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in fake]
        real_openings = torch.stack([torch.tensor(img, device=device) for img in real_openings])
        fake_openings = torch.stack([torch.tensor(img, device=device) for img in fake_openings])
        tophat_real = (real - real_openings).detach()
        tophat_fake = (fake - fake_openings).detach()
        morph_map = tophat_real ** 2 + tophat_fake ** 2

        # === SOBEL MAP (same as before)
        sobel_x = self.sobel_loss.sobel_x.to(device)
        sobel_y = self.sobel_loss.sobel_y.to(device)
        grad_real = torch.sqrt(F.conv2d(real, sobel_x, padding=1) ** 2 +
                               F.conv2d(real, sobel_y, padding=1) ** 2).detach()
        grad_fake = torch.sqrt(F.conv2d(fake, sobel_x, padding=1) ** 2 +
                               F.conv2d(fake, sobel_y, padding=1) ** 2).detach()
        sobel_map = grad_real + grad_fake

        # === Combine maps by summing them
        combined_map = morph_map + sobel_map
        
        # === Combine maps by multiplying them
        # combined_map = morph_map * sobel_map 

        # === Final weighted MSE
        diff_squared = (real - fake) ** 2
        return torch.mean(combined_map * diff_squared)
    

class SummedSpatialLossTest:
    def __init__(self, kernel_size=3):
        self.morph_loss = MorphologicalPixelLoss(kernel_size)
        self.sobel_loss = SobelPixelLoss()

    def __call__(self, fake, real):
        device = real.device

        # === MORPHOLOGICAL MAP (same as before)
        kernel = morphology.disk(self.morph_loss.kernel.shape[0] // 2)
        real_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in real]
        fake_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in fake]
        real_openings = torch.stack([torch.tensor(img, device=device) for img in real_openings])
        fake_openings = torch.stack([torch.tensor(img, device=device) for img in fake_openings])
        tophat_real = (real - real_openings).detach()
        tophat_fake = (fake - fake_openings).detach()
        morph_map = tophat_real ** 2 + tophat_fake ** 2

        # === SOBEL MAP (same as before)
        sobel_x = self.sobel_loss.sobel_x.to(device)
        sobel_y = self.sobel_loss.sobel_y.to(device)
        grad_real = torch.sqrt(F.conv2d(real, sobel_x, padding=1) ** 2 +
                               F.conv2d(real, sobel_y, padding=1) ** 2).detach()
        grad_fake = torch.sqrt(F.conv2d(fake, sobel_x, padding=1) ** 2 +
                               F.conv2d(fake, sobel_y, padding=1) ** 2).detach()
        sobel_map = grad_real + grad_fake

        # === Combine maps by summing them
        combined_map = morph_map + sobel_map

        # === Combine maps by multiplying them
        # combined_map = morph_map * sobel_map

        # === Final weighted MSE
        diff_squared = (real - fake) ** 2
        return torch.mean(torch.sum(combined_map * diff_squared)) 


class MultipliedSpatialLossNormalized:
    def __init__(self, kernel_size=3):
        self.morph_loss = MorphologicalPixelLoss(kernel_size)
        self.sobel_loss = SobelPixelLoss()

    def __call__(self, fake, real):
        device = real.device

        # === MORPHOLOGICAL MAP (same as before)
        kernel = morphology.disk(self.morph_loss.kernel.shape[0] // 2)
        real_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in real]
        fake_openings = [morphology.opening(img.squeeze().detach().cpu().numpy(), kernel) for img in fake]
        real_openings = torch.stack([torch.tensor(img, device=device) for img in real_openings])
        fake_openings = torch.stack([torch.tensor(img, device=device) for img in fake_openings])
        tophat_real = (real - real_openings).detach()
        tophat_fake = (fake - fake_openings).detach()
        morph_map = tophat_real ** 2 + tophat_fake ** 2

        # === SOBEL MAP (same as before)
        sobel_x = self.sobel_loss.sobel_x.to(device)
        sobel_y = self.sobel_loss.sobel_y.to(device)
        grad_real = torch.sqrt(F.conv2d(real, sobel_x, padding=1) ** 2 +
                               F.conv2d(real, sobel_y, padding=1) ** 2).detach()
        grad_fake = torch.sqrt(F.conv2d(fake, sobel_x, padding=1) ** 2 +
                               F.conv2d(fake, sobel_y, padding=1) ** 2).detach()
        sobel_map = grad_real + grad_fake

        # === Combine maps by multiplying them
        combined_map = morph_map * sobel_map
        # combined_map = morph_map + sobel_map

        # === Normalize the combined map
        # combined_map = combined_map - combined_map.min()
        # combined_map = combined_map / (combined_map.max() + 1e-8)

        # === Other normalization step
        combined_map = (combined_map - combined_map.mean()) / (combined_map.std() + 1e-8)

        # === Ensure the combined map is not zero
        combined_map = torch.clamp(combined_map, min=1e-8)



        # === Final weighted MSE
        diff_squared = (real - fake) ** 2
        return torch.mean(torch.sum(combined_map * diff_squared)) 
