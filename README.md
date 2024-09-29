# Autoencoder MNIST

## Project Overview

This project focuses on developing an autoencoder model for dimensionality reduction, converting image data from a size of 784 (28x28 pixels) down to a latent representation of 128 dimensions. The quality of the images reconstructed by the decoder was assessed using the Structural Similarity Index (SSIM) on a test dataset. Two distinct models were created, and hyperparameter tuning was performed to enhance their performance. The optimal model was selected based on the SSIM evaluation results.

## Dataset

The dataset utilized is the Fashion MNIST dataset, available in `keras.dataset.fashion_mnist`. This dataset comprises 10 categories of fashion items, with each category containing 7,000 image samples.

## Preprocessing

Data preprocessing was essential for preparing the images for model training. The following steps were undertaken:

1. **Scaling**: The pixel values of the images were scaled to a range suitable for the neural network, enhancing the training stability and convergence speed.
  
2. **Channel Dimension Addition**: Since the images are in grayscale, an additional dimension was introduced to represent the channel, transforming each image from shape `(28, 28)` to `(28, 28, 1)`.

3. **Dataset Splitting**: The dataset was divided into three subsets:
   - **Training Set**: 80% (56,000 images) was allocated for training the model.
   - **Validation Set**: 10% (7,000 images) was reserved for validating model performance during training.
   - **Test Set**: 10% (7,000 images) was set aside for evaluating the model after training.

## Model Architecture

### 1. Baseline Autoencoder
- The baseline architecture consists of an encoder and a decoder designed for processing single-channel images of size 28x28 pixels.
  
**Encoder**:
  - Input layer
  - Two Conv2D layers with 32 filters, 3x3 kernel size, and ReLU activation
  - MaxPooling2D layer for downsampling
  - Flatten layer followed by a Dense layer with a latent dimension of 128
  
**Decoder**:
  - Dense layer with 6,272 units (corresponding to the flattened output)
  - Reshaping into a 14x14x32 size
  - UpSampling2D layer followed by two Conv2D layers with 32 filters and a final Conv2D layer with one filter and sigmoid activation for image reconstruction.

- The model was compiled using the Adam optimizer and Mean Squared Error (MSE) loss, with early stopping implemented to mitigate overfitting.

### 2. Modified Autoencoder
- The modified architecture improves upon the baseline model by adding more convolutional layers and convolutional transpose layers, enhancing the model's ability to capture spatial structures and intricate details in the images.

**Encoder**:
  - Three Conv2D layers with 32, 64, and 128 filters
  - Flatten layer followed by a Dense layer to produce a latent representation of dimension 128
  
**Decoder**:
  - The latent vector is transformed back into image form using a Dense layer, a Reshape layer, and three Conv2DTranspose layers with 128, 64, and 1 filters, concluding with a sigmoid activation function.

- The model was compiled with the Adam optimizer and Mean Squared Error loss to minimize discrepancies between the original and reconstructed images.

This modified model achieved an SSIM score that improved by nearly 2% compared to the baseline. As a result, this architecture was selected for manual hyperparameter tuning to determine the most effective hyperparameters.


## Evaluation Results

<img width="705" alt="image" src="https://github.com/user-attachments/assets/0afcfd87-bbf5-4dae-b70a-4b76b7ec0385">

The evaluation of the two models using SSIM (Structural Similarity Index) revealed significant improvements:

- **Baseline Model**: SSIM score of **0.8991**
- **Modified Model**: SSIM score of **0.9180**

These results indicate that the modified model provides superior image reconstruction quality, closely resembling the original images.

## Conclusion

The modifications to the autoencoder architecture, particularly the inclusion of additional convolutional layers, resulted in enhanced performance over the baseline model. The modified autoencoder achieved a higher SSIM score, demonstrating its capability to deliver more accurate dimensionality reduction representations.

In summary, the modified autoencoder is slightly more effective and optimal for achieving dimensionality reduction compared to the baseline model. Future work may involve further hyperparameter tuning and exploration of alternative architectures to improve performance even further.
