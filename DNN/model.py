import torch
from torch import nn
from torch.optim import Optimizer

from preprocessing import Encoding


class Model(nn.ModuleList):
    """
    A simple non-linear feedforward DNN

    Attributes
    ----------
    encoding : Encoding
        An encoding from input symbols to neural activities
    input_size : int
        The input dimension
    output_size : int
        The output dimension
    hidden_dim : int
        The number of hidden neurons per hidden layer
    n_hid_layers : int
        The number of hidden layers
    device : device
        The device to put the model on
    init_std : float, default 1
        The standard deviation of the initial weight distribution
    """

    def __init__(
        self,
        encoding: Encoding,
        input_size: int,
        output_size: int,
        hidden_dim: int,
        n_hid_layers: int,
        device: torch.device,
        init_std: float = 1,
    ):
        super(Model, self).__init__()

        self.device = device
        self.non_linearity = torch.nn.functional.leaky_relu
        # self.non_linearity = torch.nn.functional.sigmoid
        # self.non_linearity = lambda x: torch.nn.functional.tanh(x)

        self.encoding = encoding

        # Defining the layers
        for n in range(n_hid_layers + 1):
            dim_in, dim_out = hidden_dim, hidden_dim
            if n == 0:
                dim_in = input_size
            if n == n_hid_layers:
                dim_out = output_size

            self.append(nn.Linear(dim_in, dim_out, bias=True))

        # Initialize the parameters
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight, gain=init_std)
                nn.init.zeros_(mod.bias)

        self.to(device)

    def forward(self, x):
        a = x
        activations = []
        for n, layer in enumerate(self):
            a = layer(a)
            if n != len(self):
                a = self.non_linearity(a)
            activations.append(a)

        output = activations.pop()

        return output, activations

    def train_step(self, optimizer: Optimizer, criterion, dataloader):
        self.train()
        av_loss = 0
        for batch in dataloader:
            inputs, outputs = batch
            optimizer.zero_grad()

            output, activations = self(inputs)

            loss = criterion(torch.squeeze(output), torch.squeeze(outputs))

            loss.backward()
            optimizer.step()
            av_loss += loss / len(dataloader)
        return av_loss
