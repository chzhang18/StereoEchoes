from models.gwcnet import GwcNet_G, GwcNet_GC, StereoEchoesModel
from models.loss import depth_model_loss

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-gc": GwcNet_GC,
    "stereoechoes": StereoEchoesModel
}
