import os
import torch
# from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
#     Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
#     Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, Mamba
from models import MLP, LSTM_residual, Transformer, LSTM, Transformer_ed, UNet, LSTM_Encoder, ResNet, UNet_LSTM, CNN, \
    CNN_LSTM, CNN_LSTM_attention, LSTM_new, LSTM_attention, LSTM_attention_CNN, UNet_2D, Patch_LSTM


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # self.model_dict = {
        #     'TimesNet': TimesNet,
        #     'Autoformer': Autoformer,
        #     'Transformer': Transformer,
        #     'Nonstationary_Transformer': Nonstationary_Transformer,
        #     'DLinear': DLinear,
        #     'FEDformer': FEDformer,
        #     'Informer': Informer,
        #     'LightTS': LightTS,
        #     'Reformer': Reformer,
        #     'ETSformer': ETSformer,
        #     'PatchTST': PatchTST,
        #     'Pyraformer': Pyraformer,
        #     'MICN': MICN,
        #     'Crossformer': Crossformer,
        #     'FiLM': FiLM,
        #     'iTransformer': iTransformer,
        #     'Koopa': Koopa,
        #     'TiDE': TiDE,
        #     'FreTS': FreTS,
        #     'MambaSimple': MambaSimple,
        #     'Mamba': Mamba,
        #     'TimeMixer': TimeMixer,
        #     'TSMixer': TSMixer,
        #     'SegRNN': SegRNN, 
        # }
        self.model_dict = {
            'MLP': MLP, 
            'Transformer': Transformer, 
            'LSTM': LSTM, 
            'UNet': UNet, 
            'LSTM_Encoder': LSTM_Encoder, 
            'ResNet': ResNet, 
            'UNet_LSTM': UNet_LSTM, 
            'CNN': CNN, 
            'CNN_LSTM': CNN_LSTM, 
            'CNN_LSTM_attention': CNN_LSTM_attention, 
            'Transformer_ed': Transformer_ed, 
            'LSTM_new': LSTM_new, 
            'LSTM_attention': LSTM_attention, 
            'LSTM_attention_CNN': LSTM_attention_CNN, 
            'LSTM_residual': LSTM_residual,
            'UNet_2D': UNet_2D, 
            'Patch_LSTM': Patch_LSTM
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
