from turtle import forward
from typing import Tuple
from unicodedata import bidirectional
from webbrowser import get
import torch
import torch.nn as nn

from utils import get_device

device = get_device()

# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# https://medium.com/unit8-machine-learning-publication/transfer-learning-for-time-series-forecasting-87f39e375278
# https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7
# https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
# https://www.crosstab.io/articles/time-series-pytorch-lstm
# https://www.kaggle.com/code/omershect/learning-pytorch-lstm-deep-learning-with-m5-data/notebook
# https://github.com/pytorch/examples/blob/main/time_sequence_prediction/train.py


class LSTM(nn.Module):

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, seq_length: int, bidirectional: bool = False) -> None:
        super(LSTM, self).__init__()
        self.num_classes: int = num_classes
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.seq_length: int = seq_length

        self.init_linear = nn.Linear(
            self.input_size, self.input_size).to(device)

        self.bidirectional = bidirectional
        self.bidirectional_mult: int = 2 if self.bidirectional else 1

        # long-short term memory layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=self.bidirectional,
            batch_first=True,
        ).to(device)

        self.dropout = nn.Dropout(0.4)

        # first fully connected layer
        self.fc_1 = nn.Linear(
            hidden_size * self.bidirectional_mult, 512).to(device)
        # second fully connected layer
        self.fc_2 = nn.Linear(512, 256).to(device)
        # thrid fully connected layer
        self.fc_3 = nn.Linear(256, num_classes).to(device)
        # activation function
        self.relu = nn.LeakyReLU().to(device)

    def forward(self, input: torch.Tensor):
        # Propagate input through LSTM
        _, (hn, _) = self.lstm(input, self.get_hidden_internal_state(input))

        # Reshaping the data for the Dense layer
        out = hn.view(-1, self.hidden_size * self.bidirectional_mult)
        # out = self.relu(hn[0])
        out = self.fc_1(out)
        # out = self.dropout(out)
        out = self.relu(out)

        out = self.fc_2(out)
        # out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_3(out)

        return out

    def get_hidden_internal_state(self, input: torch.Tensor):
        hidden_state = torch.zeros(self.bidirectional_mult, input.size(
            0), self.hidden_size).requires_grad_().to(device)
        internal_state = torch.zeros(self.bidirectional_mult, input.size(
            0), self.hidden_size).requires_grad_().to(device)

        return (hidden_state, internal_state)


class UtilizationLSTM(nn.Module):

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int = 1, future: int = 0) -> None:
        super(UtilizationLSTM, self).__init__()
        self.num_classes: int = num_classes
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers = num_layers
        self.future: int = future

        # long-short term memory layer to predict cpu usage
        self.cpu_lstm = nn.LSTM(
            input_size=input_size - 2,
            hidden_size=hidden_size,
            num_layers=self.num_layers, 
            batch_first=True,
        ).to(device)

        # long-short term memory layer to predict memory usage
        self.mem_lstm = nn.LSTM(
            input_size=input_size - 2,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        ).to(device)

        self.cpu_lstm_seq = self.init_sequential_layer(hidden_size)
        self.mem_lstm_seq = self.init_sequential_layer(hidden_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cpu_input, mem_input = self.split_input(input)

        # Propagate input through LSTM
        _, (cpu_h1, _) = self.cpu_lstm(cpu_input,
                                       self.get_hidden_internal_state(input))
        _, (mem_h1, _) = self.mem_lstm(mem_input,
                                       self.get_hidden_internal_state(input))

        # Reshaping the data for the Dense layer
        cpu_h1 = cpu_h1.view(-1, self.hidden_size)
        mem_h1 = mem_h1.view(-1, self.hidden_size)

        cpu_out: torch.Tensor = self.cpu_lstm_seq(cpu_h1)
        mem_out: torch.Tensor = self.mem_lstm_seq(mem_h1)

        # Concat the two tensors column-wise
        output = torch.cat([cpu_out, mem_out], dim=1)
        output = output[(self.num_layers - 1) * input.size(0):]

        return output

    def get_hidden_internal_state(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_().to(device)
        internal_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_().to(device)

        return (hidden_state, internal_state)

    def split_input(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cpu_input, mem_input = input[:, :, 0:2], input[:, :, 2:]

        return (cpu_input, mem_input)

    def init_sequential_layer(self, hidden_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size // 2),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size // 4),

            nn.Linear(hidden_size // 4, 1)
        )


if __name__ == '__main__':

    # test_tensor = torch.ones([300, 1, 19], dtype=torch.float32).to(device)
    test_tensor = torch.ones([1000, 1, 4], dtype=torch.float32).to(device)

    # number of features
    input_size: int = test_tensor.shape[2]
    # number of features in hidden state
    hidden_size: int = test_tensor.shape[2] * 400
    # number of stacked lstm layers
    num_layers: int = 3
    # number of output classes
    num_classes: int = 1

    # lstm = LSTM(num_classes, input_size, hidden_size, num_layers, test_tensor.shape[1], bidirectional=True)

    # lstm.forward(test_tensor)

    lstm = UtilizationLSTM(num_classes, input_size, hidden_size, num_layers=3)
    print(lstm(test_tensor).shape)
