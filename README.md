# 🐶🐱 Dog vs Cat Image Classification using Custom CNN

A deep learning image-classification project that identifies whether an input image contains a **Dog** or a **Cat**.

This project uses a **Custom Convolutional Neural Network (CNN) built from scratch using PyTorch**. No pretrained CNN model such as ResNet, VGG, or MobileNet was used.

The model was trained using an **NVIDIA GeForce RTX 3050 6GB Laptop GPU** with CUDA acceleration.

---

## 📌 Project Overview

- **Project Type:** Binary Image Classification
- **Classes:** Cat and Dog
- **Model:** Custom CNN
- **Framework:** PyTorch
- **Pretrained Model:** Not used
- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Acceleration:** CUDA
- **Python:** 3.12
- **Final Dataset:** 18,301 images

The complete workflow is:

**Dataset → Data Cleaning → Preprocessing → Data Augmentation → CNN → Training → Validation → Early Stopping → Model Saving → Testing → Prediction**

---

## 🛠️ Technologies Used

- Python 3.12
- PyTorch
- Torchvision
- Pillow
- NumPy
- Matplotlib
- CUDA
- NVIDIA RTX 3050 6GB Laptop GPU

---

# 🚀 Step-by-Step Process

## 1. Create the Project Folder

```bat
mkdir DogCat_AI
cd DogCat_AI
```

## 2. Create a Virtual Environment

```bat
python -m venv venv
```

Activate it on Windows:

```bat
venv\Scripts\activate
```

The terminal should then show `(venv)`.

## 3. Install Required Libraries

```bat
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Install additional libraries:

```bat
python -m pip install pillow numpy matplotlib
```

## 4. Verify the NVIDIA GPU

Check the NVIDIA GPU:

```bat
nvidia-smi
```

PyTorch CUDA verification:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

Expected:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

---

# 📂 Dataset Preparation

The dataset was organized as:

```text
PetImages/
├── Cat/
└── Dog/
```

- Initial dataset:
  - 7,999 Cat images
  - 7,999 Dog images
- Additional images:
  - 1,156 Cat images
  - 1,156 Dog images
- Final dataset checked before training:
  - **18,301 images**

A small number of images may contain backgrounds, hands, or people. The dataset was cleaned where practical, but not every image was manually inspected.

---

# 🧹 Dataset Cleaning

A script called `check_images.py` was used to find corrupted or unreadable images.

Run:

```bat
python check_images.py
```

The initial dataset check found one corrupted image:

```text
Cat/666.jpg
```

It was removed with:

```bat
del "C:\Users\Hp\OneDrive\Documents\PetImages\Cat\666.jpg"
```

The dataset was checked again:

```bat
python check_images.py
```

Final check:

```text
Total files checked: 18,301
Bad images found: 0
```

---

# 🧠 Building the Custom CNN

The CNN was implemented in:

```text
model.py
```

The architecture includes:

- Convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Adaptive Average Pooling
- Fully Connected layers
- Dropout

Output classes:

```text
0 → Cat
1 → Dog
```

The model was built **from scratch**, without a pretrained image-classification network.

Syntax check:

```bat
python -m py_compile model.py
```

---

# 🔄 Image Preprocessing and Data Augmentation

Input images were resized to:

```text
128 × 128 pixels
```

Training augmentation included:

- Random horizontal flip
- Random rotation
- Color jitter

Example:

```python
transforms.Resize((128, 128))
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
transforms.ColorJitter(...)
```

Images were converted to tensors and normalized.

Data augmentation was used to expose the model to different image variations and help reduce overfitting.

---

# 📊 Train / Validation Split

The dataset was split into:

- **80% Training**
- **20% Validation**

For 18,301 images:

- **Training:** 14,640 images
- **Validation:** 3,661 images

A fixed random seed was used:

```python
torch.Generator().manual_seed(42)
```

This makes the split reproducible.

---

# ⚙️ Training Configuration

The training program was implemented in:

```text
train.py
```

Configuration:

```text
Optimizer       → Adam
Learning Rate   → 0.001
Weight Decay    → 0.0001
Loss Function   → CrossEntropyLoss
Batch Size      → 64
Maximum Epochs  → 15
Device          → CUDA
GPU             → NVIDIA RTX 3050 6GB
```

Start training:

```bat
python train.py
```

---

# 🛑 Early Stopping

Early stopping was used to help control overfitting.

- Validation accuracy was monitored after each epoch.
- If validation accuracy did not improve for **3 consecutive epochs**, training stopped.
- The best-performing checkpoint was retained.

---

# 💾 Model Checkpoint

The best model was saved as:

```text
dog_cat_model_best.pth
```

Check that the model exists:

```bat
dir dog_cat_model_best.pth
```

The `.pth` file contains the learned CNN weights and can be loaded later without retraining.

---

# 🔮 Prediction

The prediction program was implemented in:

```text
predict.py
```

Run:

```bat
python predict.py
```

The program:

1. Loads the trained model.
2. Loads an input image.
3. Resizes and normalizes the image.
4. Runs the CNN.
5. Calculates class probabilities.
6. Displays the predicted class and confidence.

---

# 🧪 Independent Testing

Four independent images were tested:

| Image | Prediction | Confidence | Result |
|---|---|---:|---|
| `test_dog1.jpg` | Dog | 83.36% | ✅ Correct |
| `test_cat1.jpg` | Dog | 52.60% | ❌ Incorrect |
| `test_dog.jpg` | Dog | 82.95% | ✅ Correct |
| `test_cat.jpg` | Cat | 79.75% | ✅ Correct |

### Test Result

- Correct predictions: **3 / 4**
- Independent test accuracy: **75%**

The `test_cat1.jpg` image was classified as Dog with 52.60% confidence.

This four-image test is small and should not be considered a statistically robust estimate of real-world model performance.

---

# 📈 Training Result

- **Best validation accuracy:** 75.28%
- **Independent test accuracy:** 75%
- **Training device:** CUDA
- **GPU:** NVIDIA RTX 3050 6GB
- **Model:** Custom CNN
- **Pretrained model:** Not used

The best validation checkpoint was saved as:

```text
dog_cat_model_best.pth
```

---

# 📁 Project Structure

```text
DogCat_AI/
│
├── PetImages/
│   ├── Cat/
│   └── Dog/
│
├── model.py
├── train.py
├── predict.py
├── check_images.py
├── dog_cat_model_best.pth
│
├── test_cat.jpg
├── test_dog.jpg
├── test_cat1.jpg
└── test_dog1.jpg
```

> Do not upload the `venv/` folder to GitHub.

---

# ▶️ How to Run the Project

Navigate to the project:

```bat
cd C:\Users\Hp\DogCat_AI
```

Activate the virtual environment:

```bat
venv\Scripts\activate
```

Check the dataset:

```bat
python check_images.py
```

Train the model:

```bat
python train.py
```

Run prediction:

```bat
python predict.py
```

---

# 🧪 Testing New Images

The current `predict.py` uses specific image filenames.

For a new image:

1. Put the image inside the project folder.
2. Add its filename to the list in `predict.py`.
3. Save the file.
4. Run:

```bat
python predict.py
```

A future version can automatically detect every image inside a `test_images/` folder so that filenames do not need to be manually added.

---

# 🔮 Future Improvements

- Improve validation accuracy.
- Add a larger and more diverse dataset.
- Improve data augmentation.
- Tune the CNN architecture.
- Tune learning rate and batch size.
- Add a dedicated test dataset.
- Automatically detect all images in a test folder.
- Create a graphical user interface.
- Build a web application for prediction.
- Compare different custom CNN architectures.
- Add training/validation accuracy and loss graphs.

---

# ⚠️ Important Notes

- This project uses a **custom CNN trained from scratch**.
- No pretrained image-classification model was used.
- Accuracy depends on the dataset, preprocessing, architecture, and test setup.
- The four-image independent test is too small to represent general real-world performance.
- The full dataset may be too large for GitHub and may have licensing or redistribution restrictions.
- Consider providing dataset instructions rather than uploading the complete dataset.
- Do not upload the `venv/` directory.

---

# 🚫 .gitignore

Create a `.gitignore` file containing:

```gitignore
venv/
__pycache__/
*.pyc
```

---

# 👨‍💻 Author

**Harish K**

AI & Data Science Student

---

## ⭐ Project Highlights

- ✅ Custom CNN built from scratch
- ✅ No pretrained model
- ✅ PyTorch
- ✅ CUDA GPU acceleration
- ✅ NVIDIA RTX 3050 6GB Laptop GPU
- ✅ 18K+ image dataset
- ✅ Dataset validation and cleaning
- ✅ Image preprocessing
- ✅ Data augmentation
- ✅ Batch Normalization
- ✅ Dropout regularization
- ✅ Early stopping
- ✅ Best-model checkpointing
- ✅ Independent image testing
- ✅ Cat vs Dog binary classification
- ✅ Complete end-to-end deep learning workflow

---

## 📌 One-Line GitHub Description

> Custom CNN-based Cat vs Dog image classification model built from scratch using PyTorch and CUDA, trained on an 18K+ image dataset with an NVIDIA RTX 3050 GPU.
