Sign Language Detection
This project is an implementation of a Sign Language Detection system that uses an Artificial Neural Network (ANN) to identify hand gestures corresponding to letters in the American Sign Language (ASL) alphabet. The project includes a Streamlit application that allows users to upload an image of a hand sign and receive a prediction of the corresponding letter.

Project Structure
Sign_Language_Detection.ipynb: This Jupyter notebook contains the code for training the ANN model on the Sign Language MNIST dataset. The notebook includes data preprocessing, model architecture, training loop, and evaluation of the model's performance.

sign_mnist_train.csv: The CSV file containing the training data for the Sign Language MNIST dataset, which consists of grayscale images of hand signs represented as 28x28 pixels.

sign_mnist_test.csv: The CSV file containing the test data for evaluating the model.

model.pth: The trained PyTorch model saved after training. This file is loaded by the Streamlit app to make predictions.

streamlit_app.py: The Streamlit application file that provides a user interface for uploading an image of a hand sign and getting the corresponding letter prediction from the model.

How to Run the Application
Clone the Repository

Install the required packages using pip:
pip install -r requirements.txt

Run the Streamlit App:

streamlit run streamlit_app.py
Use the Application:

Open the application in your browser (usually at http://localhost:8501).
Upload an image of a hand sign (in .jpg, .jpeg, or .png format).
The model will predict and display the corresponding letter from the ASL alphabet.
Model Architecture
The ANN model consists of the following layers:

Layer 1: Fully connected layer with 784 input features and 512 output features.
Layer 2: Fully connected layer with 512 input features and 256 output features.
Layer 3: Fully connected layer with 256 input features and 128 output features.
Layer 4: Fully connected layer with 128 input features and 24 output features (corresponding to 24 letters, excluding J and Z).
The model uses ReLU activation functions and dropout layers for regularization.

Dataset
The dataset used is a subset of the MNIST dataset, adapted for sign language gestures. Each image is 28x28 pixels, grayscale, and represents one of the 24 letters of the ASL alphabet.