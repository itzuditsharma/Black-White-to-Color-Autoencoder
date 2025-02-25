import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
class ColorAutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = torch.nn.Conv2d(1, 64, 3, stride=2)
        self.down2 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.up1 = torch.nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.up2 = torch.nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
        self.up3 = torch.nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = torch.nn.ConvTranspose2d(128, 3, 3, stride=2, output_padding=1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))
        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(torch.cat((u1, d3), dim=1)))
        u3 = self.relu(self.up3(torch.cat((u2, d2), dim=1)))
        u4 = self.sigmoid(self.up4(torch.cat((u3, d1), dim=1)))
        return u4

# Load model and set to evaluation mode
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ColorAutoEncoder().to(DEVICE)
model.load_state_dict(torch.load("color_autoencoder.pth", map_location=DEVICE))
model.eval()

# Streamlit UI
st.title("Black & White to Color Autoencoder")
st.write("Upload a black-and-white image, and the autoencoder will predict a colored version!")

uploaded_file = st.file_uploader("Choose a black-and-white image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    input_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input_np, cmap="gray")
    axs[0].set_title("Grayscale Input")
    axs[0].axis("off")
    
    axs[1].imshow(output_np)
    axs[1].set_title("Predicted Color Output")
    axs[1].axis("off")
    
    st.pyplot(fig)
