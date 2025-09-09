# Indian AI NFT Diffusion Project

Generate unique Indian-inspired NFT images using deep learning diffusion and VAE architectures. This project combines 10+ Indian datasets and implements a custom VAE + U-Net pipeline for generative art.

---

## 🚀 Overview

- **Goal:** Build and train a generative model to create Indian-themed NFT images using diffusion techniques.
- **Datasets:** Over 21,000 images from Indian monuments, personalities, art, currency, food, and more.
- **Status:** Training partially completed; model outputs remain noisy due to time and hardware constraints (see results).

---

## ⚡ Hardware Requirements

- **GPU (Recommended):** Training is **very GPU-intensive**; expect long runtimes without high-end hardware (NVIDIA Tesla T4 or better ideal).
- **CPU:** Training on CPU is possible but NOT recommended—could take days/weeks for image generation.[attached_file:1]
- **Memory:** At least 8GB VRAM suggested for large datasets.[attached_file:1]

---

## 📁 Dataset Preparation

Combine images from multiple Kaggle datasets and organize for PyTorch training:
```python
import os, shutil
from tqdm import tqdm
BASE_INPUT_PATH = '/kaggle/input/'
SOURCE_FOLDERS = [ ... ] # see notebook for actual folder names
DESTINATION_PATH = '/kaggle/working/combined_indian_dataset/'
```
Organizing the images in the output folder:
```python
import os, shutil
base_path = '/kaggle/working/combined_indian_dataset/'
new_class_folder_path = os.path.join(base_path, 'images')
```

---

## 🧠 Model Architectures

### Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
# ... VAE structure ...
```

### U-Net and Attention model
```python
class UNet(nn.Module):
# ... U-Net, ResidualBlock, AttentionBlock ...
```


### Training Controller

Training logic
```python
class DiffusionTrainer:
# ... train_step, sample ...
```

---

## 🛠️ Training & Sampling

Main execution block for model training, checkpointing, and sampling generated images:

Main Script:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
IMG_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 2e-5

dataset = torchvision.datasets.ImageFolder(root=DATASET_PATH, transform=transforms)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, ...)
trainer = DiffusionTrainer(...)
for epoch in range(EPOCHS):
# ... training loop, checkpoints ...
sampled_imgs = trainer.sample(num_images=4)
```

---

## 📷 Results & Limitations

Despite improved optimizer and scheduler choices, results were very **noisy** due to unfinished training and limited compute. Example output after 15 epochs:

<img width="266" height="68" alt="final_generated_sample" src="https://github.com/user-attachments/assets/c7b3e231-4831-43e7-a698-f0843972eab0" />


> **Note:** Model quality improves significantly with more training time and a powerful GPU.

---

## 📝 How to Run

1. Clone this repo and download datasets to local directories.
2. Update the dataset paths as needed.
3. Run notebook cells in order.
4. Use `pip install torch torchvision tqdm` as needed.
5. Strongly recommended: Run on a GPU-enabled machine.
6. The datasets are named in the notebook, refer kaggle datasets for the datasets.

---

## 🙏 Credits & References

- Datasets: Kaggle, various Indian public image sets.
- Model: PyTorch framework, Kaggle kernel recipes.
- Code indexed by notebook cell for traceability.

---

## 💡 Acknowledgements

Thanks to all open-source dataset contributors. Pull requests for hardware and performance improvements are welcome!

---

> **Note:** All code snippets reference the relevant cell index from the attached notebook, `ai-nft-project.ipynb`.[attached_file:1]
