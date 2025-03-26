# NYCU-Computer-Vision-2025-Spring-HW1

StudentID: 110263029
Name: 楊芊華

# Introduction
Something about your work
How to install
... How to install dependencies
Performance snapshot
A snapshot of the leaderboard


This repository contains a **PyTorch** training script (`calr_cutmix.py`) for a **100-class image classification** task. The model is based on **ResNeXt101-64×4d**, pretrained on ImageNet, and uses **CutMix** augmentation to improve generalization. The script trains for a specified number of epochs, validates on a held-out set, and finally produces predictions on a test set in `CSV` format (zipped for submission).

Key features include:

- **CutMix (α=0.3) augmentation** with probability `CUTMIX_PROB=0.3`  
- **CosineAnnealingLR** for smoother learning rate decay  
- **AdamW** optimizer (learning rate set to `2e-4`, weight decay `1e-4`)  
- **Rich data augmentations** (random rotation, color jitter, etc.)  
- **Automatic log outputs** of training/validation loss and accuracy per epoch  
- Final predictions are saved to a ZIP file containing `prediction.csv`.

---

## File Structure

- **`calr_cutmix.py`**: Main script that trains the model and generates test predictions.  
- **`data/`**: Directory containing subfolders:
  - `train/`  
  - `val/`  
  - `test/`  

Each subfolder should contain images or class-sorted directories (for `train` and `val`). The `test` directory contains unlabeled images.

## How to Use

1. **Run the Script**  
   ```bash
   python calr_cutmix.py
   ```
   - The script will automatically detect the available GPU(s).  
   - By default, it uses `device="cuda:1"` if available; you can adjust if needed.

2. **Training and Validation**  
   - The script trains for `NUM_EPOCHS = 30`.  
   - A batch size of `BATCH_SIZE = 80` is used (subject to GPU memory constraints).  
   - **CutMix** is applied with a probability of `CUTMIX_PROB = 0.3`; if a batch entry meets this condition, two images are partially combined (rand_bbox).  
   - **CosineAnnealingLR** is used for learning rate scheduling.  
   - You will see epoch-by-epoch logs showing training/validation loss and accuracy.

3. **Outputs**  
   - After the final epoch, the script loads the best model weights (based on highest validation accuracy).  
   - A test inference pass is performed on `data/test`, generating a `prediction.csv` with the format `image_name,pred_label`.  
   - This CSV is then zipped into a file named `DD_HH_MM_calr_CutMix0.3_e30.zip` in the `./kaggle/working/` directory (where `DD_HH_MM` is the timestamp).

## Performance
<img width="495" alt="Screenshot 2025-03-26 at 19 28 55" src="https://github.com/user-attachments/assets/43201dab-3d60-49c4-9e23-da2bedafad5b" />
