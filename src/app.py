import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
model.load_state_dict(torch.load("data/models/mnist_cnn.pth", map_location=device))
model.eval()

data_set = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=ToTensor()
)

image, label = data_set[42]
data = image.unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(data)  # [batch, 1, 28, 28]
    prediction = output.argmax(dim=1).item()

print(f"Prediction: {prediction}")

plt.imshow(image.squeeze(0), cmap='gray')
plt.title(f"Predicted: {prediction}, True: {label}")
plt.axis('off')
plt.show()