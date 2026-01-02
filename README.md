## Overview
This project implements a simple MLP classifier using PyTorch on the Fashion-MNIST dataset.
The goal is to build a complete and runnable training pipeline, including training and evaluation.

## Training
- Dataset
- Fashion-MNIST
- Input shape: (1, 28, 28)
- Number of classes: 10

- Model
A simple MLP model implemented with `nn.Sequential`:
- Flatten
- Linear (784 → 512) + ReLU
- Linear (512 → 512) + ReLU
- Linear (512 → 10)


- Loss
CrossEntropyLoss
- Optimizer
SGD (learning rate = LR)

## Results
- Train accuracy (Epoch 10): 0.7237
- Validation accuracy: 0.7180

## Training
The training script tracks validation accuracy and saves the best-performing model checkpoint.

## Notes
During development, additional experiments were conducted to understand how the training
pipeline behaves under invalid input shapes or corrupted data.
These experiments helped clarify the importance of input contracts in ML engineering.

