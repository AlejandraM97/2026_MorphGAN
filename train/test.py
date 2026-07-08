import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error
from utils.losses import l2_loss, l2_z_loss, ssim_loss, MorphologicalLoss, SummedSpatialLoss, MorphologicalLossTest, SobelLoss, SobelLossTest, CombinedSpatialLoss, SummedSpatialLossTest, CombinedSpatialLossTest, MultipliedSpatialLossNormalized 

from config.options import Options
from config import registry
opt = Options().parse()
registry.opt = opt

# Configure logging
log_path = './results/logs'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_path, 'testing.log')),
        logging.StreamHandler()
    ]
)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------
# 1. Helper Functions
# -----------------------

def calculate_mse(input1: np.ndarray, input2: np.ndarray) -> float:
    """Calculates Mean Squared Error between two arrays."""
    return np.mean((input1 - input2) ** 2)

def prepare_data(inputs, targets, device):
    """Moves data to the appropriate device and detaches for evaluation."""
    return inputs.to(device).detach(), targets.to(device).detach()

def plot_curve(x, y, label, title, xlabel, ylabel, path):
    """Plots and saves a curve."""
    plt.figure()
    plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(path)
    plt.close()

def log_evaluation_results(auc_mse, auc_latent, auc_sum):
    """Logs evaluation results for AUC scores."""
    logging.info(f"AUC (MSE): {auc_mse:.4f}")
    logging.info(f"AUC (Latent): {auc_latent:.4f}")
    logging.info(f"AUC (Sum): {auc_sum:.4f}")

# -----------------------
# 2. GANTester Class
# -----------------------

class GANTester:
    """
    Tests the GANLung model for anomaly detection.

    Args:
        model (nn.Module): Trained GAN model.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Testing data loader.
        device (str): Device to run the model on.
    """
    def __init__(self, model, train_loader, test_loader, device: str) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.real_images, self.generated_images = [], []
        self.latent_inputs, self.latent_outputs, self.latent_distances = [], [], []
        self.labels, self.scores = [], []
        self.alpha = 0.5  # Weight for MSE
        self.combination_fn = self.default_combination
        self.opt = registry.opt
        

    def calculate_mse_metrics(real_img, fake_img, latent_input, latent_output, loss_type='mse'):
        mse = ssim(real_img, fake_img).item() if loss_type == 'ssmi' else mean_squared_error(real_img, fake_img)
        latent_mse = mean_squared_error(latent_input, latent_output)
        return mse, latent_mse, mse + latent_mse

    def default_combination(self, mse, latent):
        return self.alpha * mse + (1 - self.alpha) * latent
    
    def test(self):
        self.model.generator.eval()
        self.model.discriminator.eval()

        metrics = {
            0: {"mse": [], "latent": [], "sum": []},
            1: {"mse": [], "latent": [], "sum": []}
        }

        all_true_labels = []

        all_scores = {
            "mse": [],
            "latent": [],
            "sum": []
        }

        alpha = 0.9  # Weight MSE 

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                fake, latent_i, latent_o = self.model.forward_g(inputs)

                inputs_np = inputs.cpu().numpy()
                fake_np = fake.cpu().numpy()
                latent_i_np = latent_i.cpu().numpy()
                latent_o_np = latent_o.cpu().numpy()
                targets_np = targets.cpu().numpy()

                for j in range(inputs.size(0)):
                    real_img = inputs_np[j] #.flatten()
                    fake_img = fake_np[j] #.flatten()
                    latent_input = latent_i_np[j].flatten()
                    latent_output = latent_o_np[j].flatten()

                    latent_dist = np.linalg.norm(latent_input - latent_output)
                    # latent_dist = np.mean((latent_input - latent_output) ** 2)

                    if self.opt.con_loss == 'ssmi':
                        mse = ssim(real_img, fake_img).item()
                    else:
                        # For base and latent stats methodologies
                        mse = mean_squared_error(real_img.flatten(), fake_img.flatten())
                        # For morphological losses
                        # morph_loss = MorphologicalLossTest()
                        # mse = morph_loss(fake_img, real_img).cpu().numpy()
                        # For sobel losses
                        # sobel_loss = SobelLossTest()
                        # mse = sobel_loss(fake_img, real_img).cpu().numpy()
                        # For combined spatial losses
                        # combined_loss = CombinedSpatialLossTest()
                        # combined_loss = combined_loss(fake, inputs).cpu().numpy()
                        # For summed spatial losses
                        # summed_loss = SummedSpatialLossTest()
                        # mse = summed_loss(fake, inputs).cpu().numpy() 
                        # Multiplied loss
                        # multiplied_loss = MultipliedSpatialLossNormalized()
                        # mse = multiplied_loss(fake, inputs).cpu().numpy() 


                        # MSEponderado(sobel + morph)  + mse....
                        # mse = morph_loss + mse_loss
                        # mse = (sobel_loss + morph_loss) #+ mse_loss
                        # mse = ((0.5*sobel_loss) * (0.5*morph_loss)) 
                        # mse = sobel_loss + combined_loss


                    label = targets_np[j]
                    all_true_labels.append(label)

                    # Save individual scores
                    all_scores["mse"].append(mse)
                    all_scores["latent"].append(latent_dist)
                    combined_score = self.combination_fn(mse, latent_dist)
                    all_scores["sum"].append(combined_score)
                    # all_scores["image_real"].append(inputs_np[j])
                    # all_scores["image_generated"].append(fake_np[j])

                    # Store per-class for metrics
                    metrics[label]["mse"].append(mse)
                    metrics[label]["latent"].append(latent_dist)
                    metrics[label]["sum"].append(combined_score)

                    # Optional: Save for inspection
                    self.real_images.append(inputs_np[j])
                    self.generated_images.append(fake_np[j])
                    self.latent_inputs.append(latent_i_np[j])
                    self.latent_outputs.append(latent_o_np[j])
                    self.labels.append(label)

        return metrics, all_true_labels, all_scores

    
    def get_threshold_precision_recall(self, train_loader):
        """
        Computes thresholds from the train subset based on best F1 score.
        Returns a dictionary like:
            {
                'mse': threshold_mse,
                'latent': threshold_latent,
                'sum': threshold_sum
            }
        """
        self.model.generator.eval()
        self.model.discriminator.eval()

        mse_list, latent_list, sum_list, true_labels = [], [], [], []

        alpha = 0.9  # Must match the one in test()

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                fake, latent_i, latent_o = self.model.forward_g(inputs)

                for real, fake_img, lat_i, lat_o, label in zip(inputs, fake, latent_i, latent_o, labels):
                    real_img = real.cpu().numpy()#.flatten()
                    fake_img = fake_img.cpu().numpy()#.flatten()
                    latent_input = lat_i.cpu().numpy().flatten()
                    latent_output = lat_o.cpu().numpy().flatten()

                    if self.opt.con_loss == 'ssmi':
                        mse = ssim(real_img, fake_img).item()
                    else:
                        # For base and latent stats methodologies
                        mse = mean_squared_error(real_img.flatten(), fake_img.flatten())
                        # For morphological losses
                        # morph_loss = MorphologicalLossTest()
                        # mse = morph_loss(fake_img, real_img).cpu().numpy()
                        # For sobel losses
                        # sobel_loss = SobelLossTest()
                        # mse = sobel_loss(fake_img, real_img).cpu().numpy()
                        # For combined spatial losses
                        # combined_loss = CombinedSpatialLossTest()
                        # combined_loss = combined_loss(fake, inputs).cpu().numpy()
                        # For summed spatial losses
                        # summed_loss = SummedSpatialLossTest()
                        # mse = summed_loss(fake, inputs).cpu().numpy() 
                        # Multiplied loss
                        # multiplied_loss = MultipliedSpatialLossNormalized()
                        # mse = multiplied_loss(fake, inputs).cpu().numpy() 


                        # MSEponderado(sobel + morph)  + mse....
                        # mse = morph_loss + mse_loss
                        # mse = (sobel_loss + morph_loss) #+ mse_loss
                        # mse = ((0.5*sobel_loss) * (0.5*morph_loss)) 
                        # mse = sobel_loss + combined_loss

                    latent_dist = np.linalg.norm(latent_input - latent_output)
                    # latent_dist = np.mean((latent_input - latent_output) ** 2)
                    combined_score = self.combination_fn(mse, latent_dist)

                    mse_list.append(mse)
                    latent_list.append(latent_dist)
                    sum_list.append(combined_score)
                    true_labels.append(label.item())

        mse_list = np.array(mse_list)
        latent_list = np.array(latent_list)
        sum_list = np.array(sum_list)
        true_labels = np.array(true_labels)

        def find_best_threshold(metric):
            precision, recall, thresholds = precision_recall_curve(true_labels, metric)
            # print(f'true_labels: {true_labels}, {np.unique(true_labels)}')
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            return thresholds[best_idx], precision, recall, thresholds

        threshold_mse, _, _, _ = find_best_threshold(mse_list)
        threshold_latent, _, _, _ = find_best_threshold(latent_list)
        threshold_sum, _, _, _ = find_best_threshold(sum_list)

        return {
            "mse": threshold_mse,
            "latent": threshold_latent,
            "sum": threshold_sum
        }

    from sklearn.metrics import roc_auc_score
    def calculate_classification_metrics(self, thresholds, metrics, y_true=None, scores=None):
        results = {}
        preds = {}

        # No need to map anymore since your keys are consistent
        metric_mapping = {
            'mse': 'mse',
            'latent': 'latent',
            'sum': 'sum'
        }
        score_mapping = {
            'mse': 'mse',
            'latent': 'latent',
            'sum': 'sum'
        }


        for key, threshold in thresholds.items():
            metric_key = metric_mapping[key]
            score_key = score_mapping[key]

            label0_values = metrics[0][metric_key]
            label1_values = metrics[1][metric_key]

            TN = sum(1 for val in label0_values if val < threshold)
            FP = sum(1 for val in label0_values if val >= threshold)
            FN = sum(1 for val in label1_values if val < threshold)
            TP = sum(1 for val in label1_values if val >= threshold)

            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            auc_score = None
            y_pred = None
            if scores is not None and y_true is not None:
                try:
                    selected_scores = np.array(scores[score_key])  # ✅ Correct key here now
                    auc_score = roc_auc_score(y_true, selected_scores)
                    y_pred = (selected_scores >= threshold).astype(int)
                    preds[key] = y_pred
                except Exception as e:
                    print(f"❌ Error computing AUC for '{score_key}':", e)
                    print("  selected_scores shape:", selected_scores.shape)
                    print("  y_true shape:", np.array(y_true).shape)

            results[key] = {
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TP': TP,
                'specificity': specificity,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'auc': auc_score
            }

        return results, preds
    
    def plot_pca(self):
        from sklearn.decomposition import PCA

        # Flatten the latent inputs and outputs
        latent_inputs = np.array(self.latent_inputs).reshape(len(self.latent_inputs), -1)
        latent_outputs = np.array(self.latent_outputs).reshape(len(self.latent_outputs), -1)
        labels = np.array(self.labels)

        pca_model = PCA(n_components=2)
        
        pca_latent_inputs = pca_model.fit_transform(latent_inputs)
        pca_latent_outputs = pca_model.fit_transform(latent_outputs)

        plt.figure(figsize=(16, 8))

        # Plot latent inputs
        plt.subplot(1, 2, 1)
        for label in np.unique(labels):
            plt.scatter(pca_latent_inputs[labels == label, 0], pca_latent_inputs[labels == label, 1], label=f'Class {label}', alpha=0.6)
        plt.title('PCA of Latent Inputs (z)')
        plt.legend()
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks

        # Plot latent outputs
        plt.subplot(1, 2, 2)
        for label in np.unique(labels):
            plt.scatter(pca_latent_outputs[labels == label, 0], pca_latent_outputs[labels == label, 1], label=f'Class {label}', alpha=0.6)
        plt.title('PCA of Latent Outputs (z_hat)')
        plt.legend()
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.legend()

        plt.show()

    
    def plot_tsne(self):
        from sklearn.manifold import TSNE

        # Flatten the latent inputs and outputs
        latent_inputs = np.array(self.latent_inputs).reshape(len(self.latent_inputs), -1)
        latent_outputs = np.array(self.latent_outputs).reshape(len(self.latent_outputs), -1)
        labels = np.array(self.labels)

        tsne_model = TSNE(n_components=2, random_state=42)
        
        tsne_latent_inputs = tsne_model.fit_transform(latent_inputs)
        tsne_latent_outputs = tsne_model.fit_transform(latent_outputs)

        plt.figure(figsize=(16, 8))
        print(f"tsne_latent_inputs: {len(tsne_latent_inputs)}, tsne_latent_outputs: {len(tsne_latent_outputs)}, labels: {len(labels)}")

        # Plot latent inputs
        plt.subplot(1, 2, 1)
        for label in np.unique(labels):
            plt.scatter(tsne_latent_inputs[labels == label, 0], tsne_latent_inputs[labels == label, 1], label=f'Class {label}', alpha=0.4)
        plt.title('t-SNE of Latent Inputs (z)')
        plt.legend()
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks

        # Plot latent outputs
        plt.subplot(1, 2, 2)
        for label in np.unique(labels):
            plt.scatter(tsne_latent_outputs[labels == label, 0], tsne_latent_outputs[labels == label, 1], label=f'Class {label}', alpha=0.4)
        plt.title('t-SNE of Latent Outputs (z_hat)')
        plt.legend()
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks   

        plt.show()