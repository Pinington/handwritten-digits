import torch
from model import Net  # make sure model.py is in same folder or package

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load once at import
model = Net().to(device)
model.load_state_dict(torch.load("data/models/mnist_cnn.pth", map_location=device))
model.eval()

# Dummy input matching your model input shape
dummy_input = torch.randn(1, 1, 28, 28, device=device)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "data/models/digit_model.onnx",    # output ONNX file
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}  # allows variable batch sizes
)

def predict(im_data):
    """
    im_data: torch.Tensor of shape [1, 1, 28, 28], normalized [0,1]
    returns: int prediction 0-9
    """
    with torch.no_grad():
        output = model(im_data.to(device))
        prediction = output.argmax(dim=1).item()
        return prediction
