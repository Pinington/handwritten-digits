import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.optim as optim
import torch.nn as nn
from model import Net

''' This file trains the model on 0-9 and operation symbols as well '''

LEARNING_RATE = 0.005

label_to_idx = {
    '*': 0, '+': 1, '-': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7,
    '5': 8, '6': 9, '7': 10, '8': 11, '9': 12, '[': 13, ']': 14
}

class NPYTupleDataset(Dataset):
    def __init__(self, npy_file, label_to_index):
        self.data = np.load(npy_file, allow_pickle=True)
        self.label_to_index = label_to_index
        # Filter out % label
        self.data = [(img, lbl) for img, lbl in self.data if lbl in label_to_index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.label_to_index[label], dtype=torch.long)
        return image, label
    

train_data = NPYTupleDataset("data/KAGGLE/CompleteDataSet_training_tuples.npy", label_to_idx)
test_data  = NPYTupleDataset("data/KAGGLE/CompleteDataSet_testing_tuples.npy", label_to_idx)

loaders = {
    'train': DataLoader(train_data,
                        batch_size=100,
                        shuffle=True,
                        num_workers=1),

    'test': DataLoader(test_data,
                       batch_size=100,
                       shuffle=False,
                       num_workers=1)
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(15).to(device)
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
            print(f"Train Epoch: {epoch} [{batch_index * len(data)}/{len(loaders['train'].dataset)} ", end = "")
            print(f"({100. * batch_index / len(loaders['train']):.0f}%)] {loss.item():.6f}")


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
    for epoch in range(1, 9):
        train(epoch)
        test()

    torch.save(model.state_dict(), "data/models/kaggle_cnn.pth")