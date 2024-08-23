import streamlit as st
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Define the ANN model
class ANN_Model(nn.Module):
    def __init__(self):
        super(ANN_Model, self).__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 24)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Load the trained model
model = ANN_Model()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Streamlit UI
st.title("Sign Language Detection")
st.write("Upload an image of a hand sign, and the model will predict the corresponding letter.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file).convert('L')
    transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Apply the transformations
    image_tensor = transform(image)

    # Flatten the image tensor to match the input size of the ANN model (28*28 = 784)
    image_tensor = image_tensor.view(-1, 28 * 28)  # Flatten the tensor from [1, 28, 28] to [1, 784]
    
    # Convert the image to a numpy array
    image_array = np.array(image).astype(np.float32).flatten().reshape(1, -1)


    # Make sure the model is in evaluation mode
    model.eval()

   
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    class_id = predicted.item()

    # Map class_id to the corresponding letter
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
    predicted_letter = alphabet[class_id]

    st.write(f"Predicted Letter: {predicted_letter}")