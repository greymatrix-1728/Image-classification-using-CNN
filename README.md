# 🐱🐶 Image Classification using Convolutional Neural Network (CNN)

A deep learning project that classifies images as **cats or dogs** using a Convolutional Neural Network built with TensorFlow and Keras.

---

## 📌 Project Overview

This project implements a CNN from scratch to perform binary image classification. The model is trained on a labeled dataset of cat and dog images, applying data augmentation techniques to improve generalization, and achieves solid classification performance on unseen test images.

---

## 🗂️ Project Structure

```
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   └── dogs/
│   ├── test_set/
│   │   ├── cats/
│   │   └── dogs/
│   └── single_prediction/
│       └── cat_or_dog_1.jpg
├── convolutional_neural_network.ipynb
├── convolutional_neural_network.py
└── README.md
```

---

## 🧠 Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D | 32 filters, 3×3 kernel, ReLU, input shape (64, 64, 3) |
| MaxPooling2D | pool size 2×2, stride 2 |
| Conv2D | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | pool size 2×2, stride 2 |
| Flatten | Converts 2D feature maps to 1D |
| Dense | 128 units, ReLU activation |
| Dense (Output) | 1 unit, Sigmoid activation |

- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Epochs:** 25  

---

## 🔧 Data Preprocessing

### Training Set
- Rescaling pixel values to [0, 1]
- Shear transformation (range: 0.2)
- Zoom augmentation (range: 0.2)
- Horizontal flip

### Test Set
- Rescaling pixel values to [0, 1] only (no augmentation)

Images are resized to **64×64** pixels with a batch size of **32**.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy
```

### Run the project

```bash
# Clone the repository
git clone https://github.com/your-username/cnn-image-classification.git
cd cnn-image-classification

# Run the script
python convolutional_neural_network.py
```

Or open `convolutional_neural_network.ipynb` in **Google Colab** or **Jupyter Notebook**.

---

## 🔍 Making a Single Prediction

```python
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
prediction = 'dog' if result[0][0] == 1 else 'cat'
print(prediction)
```

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
