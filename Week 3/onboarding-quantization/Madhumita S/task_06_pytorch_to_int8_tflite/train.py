import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from model_definition import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])

# Downloading MNIST dataset
train_data = datasets.MNIST(".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\data",train=True,download=True,transform=transform)
test_data = datasets.MNIST(".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\data",train=False,download=True,transform=transform)

# Loading images in batches
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

#Creating the model
model = SimpleCNN()
model.to(device)
# Defining the loss function
criterion = nn.CrossEntropyLoss()
# Choosing optimizer
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Creating training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss = {running_loss/len(train_loader):.4f}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
accuracy = 100*correct/total
print(f"Accuracy = {accuracy:.2f}%")

torch.save(model.state_dict(),"onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\model.pth")
print("Model Saved Successfully")

import os
import numpy as np

os.makedirs(".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\calib", exist_ok=True)

for i in range(50):
    sample, _ = test_data[i]
    np.save(f".\\onboarding-quantization\\Madhumita S\\task_06_pytorch_to_int8_tflite\\calib\\{i}.npy", sample.numpy())

print("Calibration data saved successfully!")