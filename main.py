from config import get_parameters
from transforms import registrate_NFBS, reg_IATM_controls
from train import train
from test import test

def main(config):

    if config['mode'] == 'reg':

        if config['reg_mode'] == 'nfbs':
            registrate_NFBS(config)
        elif config['reg_mode'] == 'iatm-ctrls':
            reg_IATM_controls(config)
            
    elif config['mode'] == 'train':

        train(config)

    elif config['mode'] == 'test':

        test(config)

if __name__ == '__main__':
    config = get_parameters('train')
    main(config)