import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    IMG_SIZE = 28

    def __init__(self, out):
        super(Net, self).__init__()

        conv1_out = 32
        kernel1_size = 5

        conv1_out_size = (self.IMG_SIZE - kernel1_size + 1)
        pool1_out_size = conv1_out_size // 2 # 2x2 MaxPool used so we divide output sizes by 2

        conv2_out = 64
        kernel2_size = 5
        dropout_ratio = 0.2

        conv2_out_size = (pool1_out_size - kernel2_size + 1) 
        pool2_out_size = conv2_out_size // 2 # 2x2 MaxPool used so we divide output sizes by 2

        self.fc1_in = conv2_out * pool2_out_size * pool2_out_size
        fc1_out = 128

        # Convolutional layers 
        # Stride is 1 because images are small
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=kernel1_size)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=kernel2_size)
        self.conv2_dropout = nn.Dropout2d(p = dropout_ratio)

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_in, fc1_out)
        self.fc2 = nn.Linear(fc1_out, out) # 10 digits 


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))

        x = x.view(-1, self.fc1_in)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x