# Deep_Learning_Improved_GANs

# 🌌 Enhanced Training Techniques for Generative Adversarial Networks (GANs)

This project extends the foundational research of *"Improved Techniques for Training GANs" (2016)* by applying and benchmarking multiple GAN architectures across diverse datasets. Our goal was to stabilize training, enhance image generation quality, and improve evaluation reliability using advanced techniques like Wasserstein GAN with Gradient Penalty (WGAN-GP), DCGAN, and custom regularization strategies.

## 🎯 Objectives

- Improve GAN training stability and convergence.
- Apply effective regularization and normalization methods.
- Evaluate performance using robust metrics like FID.
- Explore cross-dataset generalization from MNIST to ImageNet.

---

## 🧠 Techniques Used

- **Wasserstein Loss with Gradient Penalty**
- **Batch Normalization & Spectral Normalization**
- **DCGAN Architectures**
- **FID Score Evaluation**
- **Precision, Recall, and Accuracy Metrics**

---

## 🗃️ Datasets

| Dataset  | Description | Image Size | Samples |
|----------|-------------|------------|---------|
| MNIST    | Handwritten digits (grayscale) | 28×28 | 70,000 |
| CIFAR-10 | 10-class color images | 32×32 | 60,000 |
| SVHN     | House number digits (color) | 32×32 | 600,000 |
| ImageNet (Subset) | Pre-classified into folders using ResNet | 128×128 | 8,706 |

---
## 🔗 Dataset Download Links

### 📘 MNIST – Handwritten Digits
- Official Page: [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)
- Google-hosted Mirror (for quick load): [Download `.npz`](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
- PyTorch Access:
  ```python
  from torchvision.datasets import MNIST
  ```

---

### 🖼️ CIFAR-10 – 10-Class Image Dataset
- Official Page: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- Direct Download: [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- PyTorch Access:
  ```python
  from torchvision.datasets import CIFAR10
  ```

---

### 🔢 SVHN – Street View House Numbers
- Official Page: [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)
- Download Links:
  - [Train (train_32x32.mat)](http://ufldl.stanford.edu/housenumbers/train_32x32.mat)
  - [Test (test_32x32.mat)](http://ufldl.stanford.edu/housenumbers/test_32x32.mat)
  - [Extra (extra_32x32.mat)](http://ufldl.stanford.edu/housenumbers/extra_32x32.mat)
- PyTorch Access:
  ```python
  from torchvision.datasets import SVHN
  ```

---

### 🧠 ImageNet (Subset Used in Project)
- **Note:** Full ImageNet access requires registration.
- Official Site: [https://www.image-net.org/download-images](https://www.image-net.org/download-images)
- Alternative (Tiny ImageNet on Kaggle): [https://www.kaggle.com/datasets/csafrit2/tiny-imagenet](https://www.kaggle.com/datasets/csafrit2/tiny-imagenet)
- In this project:
  - Downloaded 8,706 high-quality images
  - Resized to 128×128
  - Categorized using a pretrained **ResNet model**
  - Organized into folders for structured GAN training

---

## 🧪 Experiments & Results

### 🔢 MNIST
- **Final FID Score:** 50.13  
- **G Loss:** 1.2–1.65  
- **D Loss:** 0.83–0.91  
- **Precision/Recall/Accuracy:** 0.751 / 0.162 / 0.554  

### 🖼️ CIFAR-10
- **Best FID (after 1000 epochs):** 63.98  
- **G Loss:** 4.22 | **D Loss:** 0.0115  
- **Precision/Recall/Accuracy:** 0.6 / 0.3 / 0.53  

### 🔢 SVHN
- **FID Score:** 393.13 *(needs more samples to improve)*
- **Precision/Recall/Accuracy:** 1.0 / 1.0 / 1.0 *(overfitting observed)*

### 🧠 ImageNet Subset
- **Preprocessed:** Resized to 128×128 and classified using pretrained ResNet
- **GAN Output Resolution:** 64×64
- **Images Generated:** 32,000 across epochs

---

## 🛠️ Tools & Frameworks

- **Python**, **PyTorch**, **torchvision**
- **torch-fidelity** (for FID)
- **Matplotlib** & **Seaborn** for visualization
- **Pretrained ResNet** (for classifying ImageNet subset)
- **Google Colab / CUDA** / MPS

---

## 🧩 Architecture Overview

### Generator
- **Input:** Latent vector (size 100 or 128)
- **Layers:** Linear or ConvTranspose2d
- **Activations:** Leaky ReLU, BatchNorm, Tanh

### Discriminator
- **Input:** Real/Fake images
- **Layers:** Linear or Conv2d + Spectral Norm
- **Output:** Validity score (0 to 1 via Sigmoid)

---

## 📈 Evaluation Metrics

- **Frechet Inception Distance (FID)**
- **Discriminator Precision, Recall, Accuracy**
- **Loss Curves Analysis**

---

## 🔮 Future Work

- Use **learning rate schedulers** for dynamic optimization
- Apply **data augmentation** to improve discriminator generalization
- Experiment with **larger subsets of ImageNet**
- Implement **progressive growing GANs**

---

## 📚 References

- [Improved Techniques for Training GANs (2016)](https://arxiv.org/abs/1606.03498)
- [Wasserstein GAN with Gradient Penalty](https://arxiv.org/abs/1704.00028)
- [DCGAN PyTorch Implementation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

---

## 👥 Contributors

- **Shylendra Sai Bangaru**
- **Sushrutha**
- **Poorna**
- **Muno**

---
