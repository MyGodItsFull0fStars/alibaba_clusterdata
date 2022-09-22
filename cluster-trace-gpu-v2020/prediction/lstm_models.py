from unicodedata import bidirectional
from webbrowser import get
import torch
import torch.nn as nn

from utils import get_device

device = get_device()

# https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
class LSTM(nn.Module):

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, seq_length: int) -> None:
        super(LSTM, self).__init__()
        self.num_classes: int = num_classes
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.seq_length: int = seq_length
        
        self.init_linear = nn.Linear(self.input_size, self.input_size).to(device)

        # long-short term memory layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        ).to(device)
    
        self.dropout = nn.Dropout(0.4)
        # self.dropout_fc_2 = nn.Dropout(0.2)

        # first fully connected layer
        self.fc_1 = nn.Linear(hidden_size * 2, 512).to(device)
        # second fully connected layer
        self.fc_2 = nn.Linear(512, 256).to(device)
        # thrid fully connected layer
        self.fc_3 = nn.Linear(256, num_classes).to(device)
        # activation function
        self.relu = nn.LeakyReLU().to(device)

    def forward(self, input: torch.Tensor):
        # hidden_state = torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(device)
        # internal_state = torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).to(device)

        linear_output = self.init_linear(input)
        linear_output = self.relu(linear_output)
        
        # Propagate input through LSTM
        # output, (hn, cn) = self.lstm(input, (hidden_state, internal_state))
        lstm_out, _ = self.lstm(linear_output)

        # Reshaping the data for the Dense layer
        lstm_out = lstm_out.view(-1, self.hidden_size * 2)
        # print(hn.shape)
        out = self.relu(lstm_out)
        out = self.fc_1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc_3(out)
        
        return torch.abs(out)

    

if __name__ == '__main__':
    
    test_tensor = torch.ones([300, 1, 19], dtype=torch.float32)
    
    # number of features
    input_size: int = test_tensor.shape[2]
    # number of features in hidden state
    hidden_size: int = test_tensor.shape[2] * 128
    # number of stacked lstm layers
    num_layers: int = 1
    # number of output classes
    num_classes: int = 1
    
    print(input_size, hidden_size)
    
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, test_tensor.shape[1])
    
    lstm.forward(test_tensor)