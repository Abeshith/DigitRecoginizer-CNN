# DigitRecoginizer-CNN

This project demonstrates how to build a digit recognition system using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The goal is to train a model that can accurately classify handwritten digits from the MNIST dataset.

### Overview

This project performs the following steps:

1. **Loading Data**: Import and load the MNIST dataset from CSV files containing pixel values and corresponding labels.

2. **Data Exploration**: Check for missing values and print basic statistics of the datasets.

3. **Data Preprocessing**: Normalize pixel values to a range of 0 to 1.

4. **Data Visualization**: Visualize sample images from the training dataset.

5. **Model Building**: Define a CNN model using the Sequential API from Keras.

6. **Model Compilation**: Compile the model with the Adam optimizer and categorical cross-entropy loss function.

7. **Model Training**: Train the model using a portion of the training dataset. Use early stopping to prevent overfitting.

8. **Model Evaluation**: Visualize the training history and evaluate the model's performance using accuracy score and confusion matrix.

9. **Confusion Matrix Visualization**: Visualize the confusion matrix to understand the model's classification performance.

### Dependencies

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn

### Conclusion

This project demonstrates the effectiveness of CNNs in digit recognition tasks. Further improvements can be made by experimenting with different network architectures, hyperparameters, and data augmentation techniques.

### Visualization Explanations

#### Sample Images from the MNIST Dataset
![Sample Images](sample_images.png)
- This visualization displays a grid of sample images from the MNIST dataset.
- Each image represents a handwritten digit ranging from 0 to 9.
- The images are grayscale and have been preprocessed for model training.

#### Training History
![Training History](training_history.png)
- The training history plot shows the change in training and validation accuracy and loss over epochs.
- It helps in understanding the model's convergence and potential overfitting or underfitting issues.
- The x-axis represents the number of epochs, while the y-axis represents accuracy and loss values.

#### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)
- The confusion matrix visualizes the model's classification performance.
- It shows the number of correct and incorrect predictions for each digit class.
- The diagonal elements represent the number of correctly classified instances for each class, while off-diagonal elements represent misclassifications.
- Brighter colors indicate higher values, highlighting areas of frequent misclassification.

### Results

The CNN model achieved an accuracy of approximately 98% on the test dataset, indicating its effectiveness in recognizing handwritten digits. The confusion matrix reveals that the model performs well across all digit classes, with few misclassifications.


# Reference
Notebook & Data : https://www.kaggle.com/code/abeshith/digitrecognizercnn/notebook </br>
</br>
Kaggle-Competition : https://www.kaggle.com/competitions/digit-recognizer
