# ✍️ Handwritten Digit Recognition using a Convolutional Neural Network (CNN)

## 🎯 Project Goal and Overview

This project implements a robust Deep Learning model to classify handwritten digits (0-9) using the famous **MNIST dataset**. The core objective was to move beyond simple feedforward networks and apply a **Convolutional Neural Network (CNN)** architecture to showcase proficiency in computer vision fundamentals.

  * **Model:** Simple Sequential CNN
  * **Framework:** TensorFlow/Keras
  * **Dataset:** MNIST (60,000 training images, 10,000 test images, 28x28 grayscale)
  * **Goal:** Achieve a test accuracy of $\geq 98\%$ on the unseen data.

-----

## ✅ Results and Key Learnings

The model was trained for **10 epochs** using the Adam optimizer. The results demonstrate the superior performance of CNNs for image tasks.

| Metric | Result |
| :--- | :--- |
| **Final Test Accuracy** | **98.54%** |
| **Final Test Loss** | 0.061 |

### Key Technical Takeaways

1.  **Normalization:** Dividing the pixel values by 255 was a critical preprocessing step. This scaled the data to the range $[0, 1]$, preventing gradient explosion and stabilizing the training process.
2.  **Feature Extraction Power:** The `Conv2D` and `MaxPooling2D` layers successfully learned hierarchical features (edges, curves, loops) automatically, leading to an accuracy gain of several percentage points compared to a basic Dense network.
3.  **Clean Labels:** We utilized the `sparse_categorical_crossentropy` loss function, which allowed us to keep the target labels (`y_train`, `y_test`) as single integers (0-9), simplifying the preprocessing step compared to one-hot encoding.
4.  **Jupyter to Script Transition:** The development process moved from an exploratory Jupyter Notebook (`notebooks/exploration.ipynb`) to a clean, production-ready script (`src/train_model.py`), demonstrating best practices for project structure.



## 💡 Model Architecture

The implemented CNN is shallow but effective, designed for maximum clarity and performance on MNIST.

| Layer Type | Filters/Units | Output Shape | Activation | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Conv2D** | 32 | (26, 26, 32) | ReLU | Detects low-level features (edges) |
| **MaxPooling2D** | (2, 2) | (13, 13, 32) | - | Reduces dimensionality & prevents overfitting |
| **Flatten** | - | (5408) | - | Prepares feature map for classification |
| **Dense** | 128 | (128) | ReLU | Hidden layer for complex pattern learning |
| **Dense** | 10 | (10) | Softmax | Output layer (probability distribution over 10 classes) |

-----

## ⚙️ How to Run This Project

### 1\. Project Structure

Your repository follows a professional structure to keep code clean and assets organized:

```
computerVision/
├── notebooks/
│   └── exploration.ipynb      # Initial development/data exploration
├── models/
│   └── mnist_cnn_model.h5     # The trained model weights
├── src/
│   └── train_model.py         # Final, clean training and evaluation script
├── README.md                  # (You are reading this file!)
├── requirements.txt           # Python dependencies
└── .gitignore
```



## 🔮 Future Work

This project serves as a strong foundation. Potential improvements for future iteration include:

  * **Data Augmentation:** Implement techniques (like slight rotations, shifts, or shears) to artificially expand the dataset and improve model robustness.
  * **Advanced Architecture:** Explore famous CNN architectures like LeNet-5 or VGG to push the accuracy even closer to human performance.
  * **Prediction Interface:** Create a small web application (e.g., using Streamlit or Flask) that allows a user to draw a digit and get a real-time prediction from the saved model.
