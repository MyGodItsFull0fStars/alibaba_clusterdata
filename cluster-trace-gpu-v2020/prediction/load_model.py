import torch
from lstm_models import UtilizationLSTM
from dataclasses import dataclass, field


@dataclass
class ModelParameters(object):
    input_size: int
    hidden_size: int
    num_layers: int = field(init=True, default=1)
    model_state: dict = field(init=True, repr=False, default_factory=dict)
    num_classes: int = field(init=False, repr=False, default=0)
    
    # def __repr__(self):
    #     return (
    #         f'''{self.__class__.__name__}\n- input size:\t{self.input_size}\n- hidden size:\t{self.hidden_size}\n- num layers:\t{self.num_layers}''')

def load_model_parameters(model_path: str) -> ModelParameters:
    assert model_path is not None
    assert len(model_path) != 0
    
    if torch.cuda.is_available():
        param = torch.load(model_path)
    else:
        param = torch.load(model_path, map_location=torch.device('cpu'))
    return ModelParameters(
        param['input_size'], 
        param['hidden_size'],
        param['num_layers'],
        param['model_state_dict'])

def load_lstm_model(params: ModelParameters, load_model_state: bool = False) -> UtilizationLSTM:
    model = UtilizationLSTM(
        params.input_size,
        params.hidden_size,
        params.num_layers
        )
    if load_model_state:
        model.load_state_dict(params.model_state)
    return model

if __name__ == '__main__':
    mp = load_model_parameters('models/small-epochs-100')
    print(mp)
    model = load_lstm_model(mp)
    
    
    
    