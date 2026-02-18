import torch
from .model import Net  # make sure model.py is in same folder or package

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load once at import
model = Net(15).to(device)
model.load_state_dict(torch.load("data/models/kaggle_cnn.pth", map_location=device))
model.eval()

def predict(im_data):
    """
    im_data: torch.Tensor of shape [1, 1, 28, 28], normalized [0,1]
    returns: str prediction ('0'-'9', '+', '*', etc.)
    """
    label_to_index = {
        '*': 0, '+': 1, '-': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7,
        '5': 8, '6': 9, '7': 10, '8': 11, '9': 12, '[': 13, ']': 14
    }
    index_to_label = {v: k for k, v in label_to_index.items()}

    with torch.no_grad():
        output = model(im_data.to(device))
        pred_index = output.argmax(dim=1).item()
        pred_label = index_to_label[pred_index]  # return symbol
        return pred_label
