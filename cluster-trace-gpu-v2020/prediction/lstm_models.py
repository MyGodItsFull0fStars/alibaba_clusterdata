import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, seq_length: int) -> None:
        super(LSTM, self).__init__()
        self.num_classes: int = num_classes
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.seq_length: int = seq_length

        # long-short term memory layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)

        # first fully connected layer
        self.fc_1 = nn.Linear(hidden_size, 256)
        # second fully connected layer
        self.fc_2 = nn.Linear(256, 128)
        # thrid fully connected layer
        self.fc_3 = nn.Linear(128, num_classes)
        # activation function
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        hidden_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        internal_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(input, (hidden_state, internal_state))
        # Reshaping the data for the Dense layer
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        
        return out
    