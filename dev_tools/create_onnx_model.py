import torch
import torch.nn as nn


class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(
            100, 50)  # Must Match the dimensions Ada implementation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x


# Create model instance
model = SimpleNet()

# Create dummy input (batch_size=1, input_features=100)
dummy_input = torch.randn(1, 100)

# Export the model
torch.onnx.export(
    model,  # model being run
    dummy_input,  # model input (or a tuple for multiple inputs)
    "model.onnx",  # where to save the model
    export_params=
    True,  # store the trained parameter weights inside the model file
    opset_version=11,  # the ONNX version to export the model to
    do_constant_folding=
    True,  # whether to execute constant folding for optimization
    input_names=['input'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={
        'input': {
            0: 'batch_size'
        },  # variable length axes
        'output': {
            0: 'batch_size'
        }
    })

print("Model exported to model.onnx")
