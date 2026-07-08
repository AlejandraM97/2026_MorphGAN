# MorphGAN
## Morphology-Driven Anomaly Learning for Pulmonary Nodule Classification in CT
### Official implementation of the paper: MorphGAN: A Morphology-Driven Anomaly Approach for Pulmonary Nodule Classification in CT


**Overview**
MorphGAN is an **anomaly detection** methodology for pulmonary nodule classification trained exclusively on benign nodules. Instead of learning a discriminative benign/malignant classifier, the model learns the benign distribution and detects malignant nodules according to their reconstruction error. The proposed method incorporates

- CBAM attention modules
- Morphology-guided reconstruction loss
- MMD latent distribution alignment
- Wasserstein adversarial training

to improve anomaly detection while reducing false positives

**Repository structure**
.
├── run.py                  # Main training script
├── train.py                # Training procedure
├── preprocessing.py        # Data preprocessing and augmentation
├── network.py              # MorphGAN architecture
├── network_skips.py        # Architecture with skip connections
├── gantest.py              # Evaluation
├── generator_test.py       # Reconstruction evaluation
├── config.json             # Hyperparameters
├── requirements.txt
└── README.md

**Requirements**
Python 3.10+
Main dependencies

``
PyTorch
torchvision
numpy
scikit-image
scikit-learn
opencv-python
pandas
SimpleITK
matplotlib
``

or simply: ``pip install -r requirements.txt``

**Dataset**

The experiments were trained and validated using the LIDC-IDRI dataset.

Download: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI

The preprocessing follows the protocol described in the paper:
- only nodules larger than 3 mm
- median malignancy value (MMV)
- class 3 removed
- only benign nodules were used during training
Augmentation:
- rotations
- horizontal flips
- vertical flips
- z-score normalization
