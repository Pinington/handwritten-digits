import torch
from src.model import Net

model = Net(15)
model.load_state_dict(torch.load("data/models/kaggle_cnn.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "data/models/digit_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    export_params=True,      # embed all weights
    do_constant_folding=True,
    dynamic_axes=None         # important: no dynamic axes
)
