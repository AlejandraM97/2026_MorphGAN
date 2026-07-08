# MorphGAN: Morphology-Driven Anomaly Learning for Pulmonary Nodule Classification in CT

Official implementation of:

> **MorphGAN: A Morphology-Driven Anomaly Approach for Pulmonary Nodule Classification in CT**
<p align="center">
<img src="Pipeline_Anomaly_Journal.pdf" width="900">
</p>

MorphGAN is an anomaly-based framework for pulmonary nodule classification on computed tomography (CT) images. The method is trained using benign pulmonary nodules to learn a reference distribution of benign morphology, texture, and density. Malignant nodules are then identified as out-of-distribution samples according to their reconstruction error and latent-space deviation.

The model extends a GANomaly-like architecture by incorporating as key contributions:

- Convolutional Block Attention Modules (CBAM)
- A weighted morphology reconstruction loss
- Latent-space consistency losses
- Maximum Mean Discrepancy (MMD) for global latent distribution alignment
- Adversarial learning for reconstruction-based anomaly detection.

The goal is to improve pulmonary nodule characterization while reducing false-positive predictions in malignancy classification.

The highlights can be expressed as:
- A Morphology-driven anomaly methodology for nodule malignancy characterization
- Combines global and local distribution learning to capture benign nodule patterns
- Validated on four public datasets that contain biopsy and radiological annotations
- Achieved 95.2\% AUC and 93.24\% of specificity, being competent with state-of-the-art
- Reduces strict reliance on malignancy labels while preserving diagnostic reliability

---

## Repository Structure

```text
.
├── config/
│   ├── config.json
│   ├── options.py
│   ├── registry.py
│   ├── pipeline_config.yaml
│   └── sweep_configs/
│
├── data/
│   ├── data.py
│   └── hist_loader.py
│
├── models/
│   ├── base/
│   ├── latstats/
│   ├── mmd/
│   ├── morph/
│   ├── morph_and_sobel/
│   ├── morph_and_sobel_add/
│   ├── morph_and_sobel_mult_normalized/
│   ├── morph_malignant/
│   ├── mse/
│   └── sobel/
│
├── scripts/
│   ├── run.py
│   ├── pipeline_runner.py
│   ├── plotting.py
│   └── sweep_lambdas.py
│
├── train/
│   ├── train.py
│   └── test.py
│
├── utils/
│   ├── losses.py
│   └── metrics.py
│
├── results/
│   ├── logs/
│   ├── models/
│   └── weights/
│
├── Dataset_construction.ipynb
├── test_notebook.ipynb
├── test_notebook_LIDC_Biopsy.ipynb
├── test_notebook_LNDb.ipynb
├── test_notebook_NLST.ipynb
├── test_notebook_multiclass.ipynb
└── README.md
```

---

## Method Overview

MorphGAN follows an anomaly-learning formulation. During training, the model learns the distribution of benign pulmonary nodules. At inference time, nodules producing larger reconstruction errors or larger deviations in the latent space are considered more likely to be malignant.

The proposed framework consists of four main components:

- **Generator:** an encoder-decoder-encoder architecture that reconstructs benign pulmonary nodules.
- **Discriminator:** encourages realistic reconstructions through adversarial learning.
- **CBAM Attention:** channel and spatial attention modules integrated into the decoder to improve morphological reconstruction.
- **Latent Distribution Alignment:** combines Mean Squared Error (MSE) and Maximum Mean Discrepancy (MMD) losses to align the latent representations of original and reconstructed nodules.

The complete objective function combines adversarial, reconstruction, morphology-guided, and latent distribution losses.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-user>/MorphGAN.git
cd MorphGAN
```

## 2. Create a Python environment

Using conda:

```bash
conda create -n morphgan python=3.10
conda activate morphgan
```

or using venv:

```bash
python -m venv morphgan_env
source morphgan_env/bin/activate
```

Windows:

```bash
morphgan_env\Scripts\activate
```

## 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise install the main libraries manually:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-image scikit-learn opencv-python scipy tqdm pyyaml hydra-core omegaconf wandb
```

---

# Data Availability

All datasets employed in this work are publicly available.

| Dataset | Description | Download |
|----------|-------------|----------|
| **LIDC-IDRI** | Main training dataset containing pulmonary nodules annotated by four radiologists with malignancy scores and morphological attributes. | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| **LNDb** | External radiological validation dataset with multi-radiologist annotations. | https://lndb.grand-challenge.org/Data/ |
| **NSCLC Radiogenomics** | External validation dataset containing histopathologically confirmed lung cancer cases. | https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/ |
| **NLST** | National Lung Screening Trial dataset used for external validation on screening CT studies. | https://www.cancerimagingarchive.net/collection/nlst/ |

The original medical images are **not redistributed** in this repository. Users must download the datasets directly from their official repositories and comply with the corresponding data usage agreements.

---

# Dataset Preparation

## LIDC-IDRI

The experiments were performed following the protocol described in the paper.

The preprocessing pipeline includes:

- nodules larger than 3 mm;
- nodules annotated by at least three radiologists;
- median malignancy value (MMV) as consensus label;
- exclusion of indeterminate nodules (malignancy score = 3);
- z-score normalization;
- data augmentation through rotations and horizontal/vertical flipping.

Only **benign nodules** are used during training.

The preprocessing scripts can be found inside the repository.

---

# Folder Structure

The project expects a folder containing the preprocessed nodule patches.

Example:

```text
dataset/
│
├── train/
│   ├── benign/
│   └── malignant/
│
└── test/
    ├── benign/
    └── malignant/
```

Please update the dataset paths inside

```
config/config.json
```

Or the corresponding configuration file.

---

# Configuration

The main configuration file is

```
config/config.json
```

Important parameters include:

- random seed
- batch size
- number of epochs
- learning rate
- reconstruction loss
- MMD weight
- model architecture
- dataset version
- folder paths

The reconstruction loss can be selected as

```json
"con_loss": "l2"
```

or

```json
"con_loss": "ssim"
```

---

# Training

The main MorphGAN experiment can be reproduced with

```bash
python scripts/run.py \
--model_name decoder_morph \
--model_type morph \
--model_architecture.encoder_cbam false \
--model_architecture.decoder_cbam true \
--w_mmd 10
```

This configuration corresponds to the MorphGAN model reported in the paper.

---

# Model Variants

Several reconstruction losses are available:

- MSE
- Morphological loss
- Sobel loss
- Morphological + Sobel
- MMD
- Latent statistics
- Base GANomaly

The desired model can be selected through

```
--model_type
```

---

# Testing

Several evaluation notebooks are provided.

Examples include:

```
test_notebook.ipynb
test_notebook_LNDb.ipynb
test_notebook_NLST.ipynb
test_notebook_LIDC_Biopsy.ipynb
```

Before running the notebooks, update:

- dataset path
- model path
- configuration file
- dataset version

---

# Anomaly Threshold

The model predicts an anomaly score according to the reconstruction error.

Pulmonary nodules are classified as

```
anomaly score < threshold  → benign

anomaly score ≥ threshold  → malignant
```

The threshold is computed over the benign training cohort by maximizing the F1-score on the Precision-Recall curve.

---

# Outputs

During training the repository stores

```
results/

    logs/

    weights/

    figures/
```

The trained weights are saved as

```
results/weights/<model_name>.pth
```

---

# Reproducibility

To reproduce the experiments reported in the paper, keep fixed

- random seed
- train/test split
- preprocessing protocol
- image size
- model configuration
- loss weights
- anomaly threshold selection strategy

Minor numerical differences may appear depending on the GPU, CUDA and PyTorch versions.

---

# Citation

If you use this repository in your research, please cite

```bibtex
@article{Moreno2026MorphGAN,
  title={MorphGAN: A Morphology-Driven Anomaly Approach for Pulmonary Nodule Classification in CT},
  author={Moreno, Alejandra and Rodríguez, Josué and Manzanera, Antoine and Rueda, Andrea and Henríquez, Héctor and Martínez, Fabio},
  journal={Pattern Analysis and Applications},
  year={2026}
}
```

Please update the citation after publication.

---

# License

This project is released for academic research purposes.

---

# Contact

**Alejandra Moreno**

Biomedical Imaging, Vision and Learning Laboratory (BIVL²ab)

Universidad Industrial de Santander

Email: alejandra.moreno@saber.uis.edu.co

For questions regarding the code or the experiments, please open an Issue in this repository.
