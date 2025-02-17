# Pneumonia Detection using CNN

## Overview

Pneumonia Detection using CNN is a deep learning project that leverages Convolutional Neural Networks (CNNs) to classify chest X-ray images as either normal or pneumonia-affected. The model is trained on a labeled dataset and utilizes advanced image processing techniques to enhance accuracy. This project aims to provide an efficient, automated tool for detecting pneumonia, aiding healthcare professionals in early diagnosis and treatment planning.

## Features

- Preprocessed chest X-ray dataset
- CNN architecture for image classification
- Model training and evaluation
- Performance metrics including accuracy, loss, and confusion matrix visualization
- Deployment-ready model for real-world applications

## Dataset

The dataset used for this project contains chest X-ray images categorized into:

- **Normal** (No pneumonia)
- **Pneumonia** (Bacterial or viral infection)

Dataset Source: [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Installation

### Prerequisites

Ensure you have Python and the required libraries installed:

```bash
pip install tensorflow keras numpy matplotlib seaborn opencv-python
```

### Clone the Repository

```bash
git clone https://github.com/Aathishwaran-003/Pneumonia-detection-using-CNN.git
cd Pneumonia-detection-using-CNN
```

## Model Architecture

The CNN model consists of:

- Convolutional Layers with ReLU activation
- MaxPooling Layers
- Fully Connected Layers (Dense)
- Softmax activation for classification

## Training the Model

Run the following command to train the model:

```bash
python train.py
```

This script will preprocess the data, train the CNN model, and save the trained model.

## Evaluation

To evaluate the model on the test dataset:

```bash
python evaluate.py
```

The script will output performance metrics such as accuracy, precision, recall, and confusion matrix.

## Predictions

You can test the model on new images using:

```bash
python predict.py --image sample_image.jpg
```

## Results

- Training Accuracy: **XX%**
- Validation Accuracy: **XX%**
- Confusion Matrix:



## Future Improvements

- Implementing Transfer Learning with pre-trained models like ResNet
- Deploying the model as a web or mobile application
- Enhancing dataset with more diverse X-ray images

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset by [Kaggle](https://www.kaggle.com/)
- Deep learning framework: TensorFlow/Keras

## Contact

For any questions or contributions, reach out via [GitHub Issues](https://github.com/Aathishwaran-003/Pneumonia-detection-using-CNN/issues).

