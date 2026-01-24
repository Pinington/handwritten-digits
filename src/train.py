from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
import torch.nn as nn
from model import Net

LEARNING_RATE = 0.005

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)

loaders = {
    'train': DataLoader(train_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=1),

    'test': DataLoader(test_data,
                       batch_size=100,
                       shuffle=True,
                       num_workers=1)
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_index, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_index % 25 == 0:
            print(f"Train Epoch: {epoch} [{batch_index * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_index / len(loaders['train']):.0f}%)] {loss.item():.6f}")


def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target) * data.size(0)
            prediction = output.argmax(dim = 1, keepdim = True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f"\nTest: Avg. Loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} \
           ({100. * correct / len(loaders['test'].dataset):.0f}%)\n")
    

if __name__=="__main__":
    for epoch in range(1, 6):
        train(epoch)
        test()

    torch.save(model.state_dict(), "src/mnist_cnn.pth")