# Black-White-to-Color-Autoencoder

This project implements a deep learning-based autoencoder to convert black-and-white images into color images using PyTorch and Streamlit.

Overview

The application utilizes a convolutional autoencoder trained to predict the color version of grayscale images. The model is deployed using Streamlit, allowing users to upload black-and-white images and receive colorized outputs in real-time.

Features

Upload a grayscale image in JPG, PNG, or JPEG format.

The autoencoder predicts and displays the colorized version of the input image.

Interactive UI powered by Streamlit.

Model inference runs on GPU (if available) for faster processing.


# Model Architecture

The autoencoder consists of:

Encoder: A series of convolutional layers that downsample the input image.

Decoder: Transposed convolution layers that reconstruct the colorized image from the compressed representation.


# Example Output

When a grayscale image is uploaded, the model predicts and displays a colorized version side-by-side.

Future Improvements

Train on a larger dataset for better colorization quality.

Implement GAN-based colorization for more realistic results.

Deploy the model using cloud services.
