from .transformer import *
from .performer import *

def build_model(config) :
    """
    builds or reloads model and transfer them to config.device
    """
    if config.model_type == 'Transformers':
        model = Transformers(config)
    elif config.model_type == 'Performers':
        model = Performers(config)

    if config.load_model:
        checkpoint = torch.load(config.model_path)
        print("=> Loading checkpoint")
        assert 'model' in list(checkpoint.keys())
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    model = model.to(config.device)
    return model
