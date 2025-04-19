# from .schnet import SchNetEncoder
# from .gin import GINEncoder
# from .edge import *
# from .coarse import *
# from .comenet import ComENetEncoder
# from .dimenetpp import DimeNetPPEncoder
# from .egnn import EGNNMixed2DEncoder
from core.torchmdnet.models.torchmd_et import TorchMD_ET


EncoderDict = {
    # "dimenetpp": DimeNetPPEncoder,
    # "egnn": EGNNMixed2DEncoder,
    # "schnet": SchNetEncoder,
    # "gin": GINEncoder,
    # "comenet": ComENetEncoder,
    "torchmd_et": TorchMD_ET,
}


def load_encoder(config, encoder_type="global_encoder"):
    # cfg = getattr(config, encoder_type)
    cfg = config
    encoder = EncoderDict[cfg['name']].from_config(cfg)
    return encoder
