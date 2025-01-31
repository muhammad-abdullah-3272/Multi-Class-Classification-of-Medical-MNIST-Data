# Multi-Class Classification of Medical MNIST Data

## Overview
This project focuses on multi-class classification using the Medical MNIST dataset. The classification is performed using **Softmax Activation** and **Hinge Loss**. The objective is to train a model to correctly classify medical images into one of three selected classes.

> **Note:** This is a **from-scratch implementation**, meaning no pre-built libraries like TensorFlow or PyTorch were used for model training. The dataset size is intentionally small as this was an experimental project. However, users can increase the dataset size as needed to enhance generalization and model performance.

## Dataset Details
The **Medical MNIST dataset** originally consists of six classes and approximately **60,000** images, each of size **64x64x3**. However, for this project, only three classes are selected, with **10 examples per class**, making a total dataset of **30 images**.

### Dataset Split
| Dataset Name     | Number of Examples |
|-----------------|------------------|
| Training Set    | 22               |
| Validation Set  | 4                |
| Test Set        | 4                |
| **Total**       | **30**           |

## Data Preprocessing
- **Pre-processing**: Three class folders were selected from the original dataset. Features and labels were extracted and concatenated.
- **Feature Scaling**: Since pixel values range from **0-255**, each feature is normalized by dividing by **255**.
- **Splitting**: The dataset is split into training (70%), validation (15%), and test (15%).

## Model Architecture
- **Input Layer**: 12,288 neurons (64x64x3 features)
- **Output Layer**: 3 neurons (for three classes)
- **Activation Function**: Softmax
- **Loss Function**: Hinge Loss & Categorical Cross-Entropy
- **Regularization**: L2 Regularization (λ = 0.03)
- **Optimization Algorithm**: Gradient Descent

## Training & Hyperparameters
| Parameter           | Value  |
|--------------------|--------|
| Learning Rate (α) | 0.001  |
| Regularization (λ)| 0.03   |
| Training Iterations | 500  |

## Model Training and Evaluation
The training process involved minimizing classification loss and optimizing parameters through gradient descent. The training and validation loss were recorded over multiple iterations.

### Final Training Results
| Dataset       | Cost Error |
|--------------|-----------|
| Training Loss | 0.035     |
| Validation Error | 0.116 |
| Test Error | 0.191 |

### Model Accuracies
| Dataset      | Accuracy |
|-------------|----------|
| Training Set  | 100%    |
| Validation Set | 100%  |
| Test Set     | 100%    |

> **Note**: Due to the small dataset size, the model performs exceptionally well. Accuracy may vary with a larger dataset.

## How to Run the Code
### Training the Model
1. Run `Training.py` to train the model.
2. Change dataset split settings in `Data_Preprocessing.py` if needed.
3. Adjust hyperparameters in `train()` function:
   ```python
   trainCost, valCost, trainAccuracy, valAccuracy = model.Train(X_train, Y_train, X_val, Y_val, lamda=0.03, alpha=0.001, epochs=300, print_every=100)
   ```
4. The trained model is then evaluated on the test set.

### Model Selection
- Run `lambda_Selection_3Layered.py` to analyze different model configurations.
- Modify `learning_rate`, `iterations`, and `lambda` values to experiment with different setups.

## Conclusion
This project demonstrates the effectiveness of Softmax activation and Hinge Loss in medical image classification. The model achieves **100% accuracy** on the small dataset, indicating a strong ability to learn from limited data. Further improvements can be explored with larger datasets and additional layers.
